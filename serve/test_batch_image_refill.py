"""Pinned batch image-reference misses: one request-local retry before output.

CPU protocol tests only. The fake native process never loads images or models.
It checks that each full-row file belongs to its request, while real StrataEngine
owns control queues, admission, cancellation and request-thread callbacks.
"""
import concurrent.futures
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock
from serve import server as S

FAKE = r'''import json,queue,struct,sys,threading,time
from pathlib import Path
log=open(sys.argv[sys.argv.index('--log')+1],'a',buffering=1)
q=queue.Queue();active={}
def reader():
 for line in sys.stdin:q.put(line.strip())
 q.put(None)
threading.Thread(target=reader,daemon=True).start()
print('INFO batch_slots=4 batch_pinned=1 img_refs=1 n_embd=8',flush=True)
print('READY 4096 stop',flush=True)
def note(**kw):log.write(json.dumps(kw)+'\n')
while True:
 try:line=q.get(timeout=.01 if active else None)
 except queue.Empty:line=''
 if line is None or line=='QUIT':break
 f=line.split()
 if f and f[0]=='HAVE':
  note(event='have');print('HAVE '+','.join('4:2:2' for _ in f[1].split(',')),flush=True)
 elif f and f[0]=='BSTOP':
  slot=int(f[1]);note(event='bstop',slot=slot)
  if slot in active:
   left,count=active.pop(slot);print('BDONE %d %d cancel 1'%(slot,count),flush=True)
 elif f and f[0] in ('BGEN','BGENI'):
  slot=int(f[1]);maximum=int(f[2]);ids=[int(x) for x in f[-1].split(',')];client=chr(ids[0]) if ids[0] in (65,66) else 'A'
  if f[0]=='BGENI':
   raw=Path(f[-2]).read_bytes()
   mode='REF:'+client if raw[:4]==struct.pack('<i',0x31525653) else 'FULL:'+client if raw[:4]==struct.pack('<i',0x31455653) else raw.decode()
  else:mode='FULL:'+client
  note(event='request',client=client,mode=mode,slot=slot)
  if mode.startswith('REF:'):
   print('ERR NEED_IMAGES 1 (reference evicted)',flush=True);continue
  if mode.startswith('AFTER:'):
   print('T 120',flush=True);print('ERR NEED_IMAGES 1 (after output)',flush=True);continue
  if mode!='FULL:'+client:
   print('ERR wrong request image file',flush=True);continue
  reply=list(('reply-'+client).encode())+[257]
  print('T %d'%reply[0],flush=True)
  print('DONE 1 %d 1 0 length 0 0 0'%len(ids),flush=True)
  print('BADM %d 1'%slot,flush=True)
  active[slot]=[reply[1:],1]
 if not line:
  for slot in list(active):
   left,count=active[slot];token=left.pop(0);count+=1
   print('BT %d %d'%(slot,token),flush=True)
   if token==257:
    print('BDONE %d %d stop 1'%(slot,count),flush=True);del active[slot]
   else:active[slot]=[left,count]
'''


class BatchImageRefill(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.directory=Path(self.tmp.name)
        script=self.directory/'engine.py';script.write_text(FAKE)
        self.log=self.directory/'protocol.jsonl'
        real=S.subprocess.Popen
        with mock.patch.object(S.subprocess,'Popen',lambda cmd,**kw:real([sys.executable,str(script),*cmd[1:]],**kw)):
            self.engine=S.StrataEngine('fake',['--batch','4','--log',str(self.log)])
        self.pool=concurrent.futures.ThreadPoolExecutor(max_workers=4)

    def tearDown(self):
        self.engine.close()
        self.pool.shutdown(wait=True)
        self.tmp.cleanup()

    def events(self):
        return [json.loads(l) for l in self.log.read_text().splitlines()] if self.log.exists() else []

    def image(self,client,mode='REF'):
        path=self.directory/(client+'.images');path.write_text(mode+':'+client);return path

    def run_request(self,client,path,callback,cancel=None,on_token=None):
        cancel=cancel or threading.Event()
        self.engine.refill=callback
        out=[]
        for token in self.engine.generate([ord(client)],32,{},cancel,embeddings=str(path) if path else None):
            if token is not None:
                out.append(token)
                if on_token:on_token(token,cancel)
        return bytes(t for t in out if t<256)

    def test_two_concurrent_callbacks_do_not_cross_image_files(self):
        barrier=threading.Barrier(2);calls=[];lock=threading.Lock()
        self.engine.refill='main-thread-sentinel'
        def request(client):
            path=self.image(client)
            def refill():
                self.assertIsNone(self.engine.refill)
                barrier.wait(timeout=2)
                with lock:calls.append(client)
                path.write_text('FULL:'+client);return str(path)
            result=self.run_request(client,path,refill)
            self.assertIsNone(self.engine.refill)
            return result
        a=self.pool.submit(request,'A');b=self.pool.submit(request,'B')
        self.assertEqual(a.result(timeout=4),b'reply-A');self.assertEqual(b.result(timeout=4),b'reply-B')
        self.assertCountEqual(calls,['A','B'])
        self.assertEqual(self.engine.refill,'main-thread-sentinel')
        for client in ('A','B'):
            modes=[e['mode'] for e in self.events() if e.get('client')==client]
            self.assertEqual(modes,['REF:'+client,'FULL:'+client])

    def test_old_owner_and_control_released_before_callback_fifo(self):
        fifo=threading.Lock();fifo.acquire();entered=threading.Event();cancel=threading.Event();got_ctl=threading.Event()
        path=self.image('A')
        def holder():
            self.assertTrue(entered.wait(2))
            try:
                self.assertTrue(self.engine.ctl.acquire(timeout=.5),'refill kept ctl while waiting FIFO')
                got_ctl.set();self.engine.ctl.release()
            finally:fifo.release()
        hold=self.pool.submit(holder)
        def refill():
            with self.engine.wlock:self.assertNotIn(cancel,getattr(self.engine,'_slot_cancel_owners',{}))
            entered.set()
            self.assertTrue(fifo.acquire(timeout=1))
            try:
                self.assertEqual(self.engine.have(['aa'],timeout=1),[(4,2,2)])
                path.write_text('FULL:A');return str(path)
            finally:fifo.release()
        request=self.pool.submit(self.run_request,'A',path,refill,cancel)
        self.assertEqual(request.result(timeout=4),b'reply-A');hold.result(timeout=2);self.assertTrue(got_ctl.is_set())

    def test_other_control_and_generation_progress_during_refill(self):
        entered=threading.Event();release=threading.Event();path=self.image('A')
        def refill():
            entered.set();self.assertTrue(release.wait(2));path.write_text('FULL:A');return str(path)
        a=self.pool.submit(self.run_request,'A',path,refill)
        self.assertTrue(entered.wait(2))
        self.assertEqual(self.engine.have(['bb'],timeout=1),[(4,2,2)])
        b=self.pool.submit(self.run_request,'B',None,None)
        self.assertEqual(b.result(timeout=2),b'reply-B')
        release.set();self.assertEqual(a.result(timeout=3),b'reply-A')
        self.assertFalse(any(e.get('mode','').startswith('FULL:A') for e in self.events() if e.get('client')=='B'))

    def test_one_retry_only_and_no_output_on_repeated_missing_images(self):
        path=self.image('A');calls=[]
        def refill():calls.append(1);return str(path)
        with self.assertRaisesRegex(ValueError,'NEED_IMAGES'):
            self.run_request('A',path,refill)
        self.assertEqual(calls,[1]);self.assertEqual(sum(e.get('event')=='request' for e in self.events()),2)

    def test_missing_images_after_output_never_retries(self):
        path=self.image('A','AFTER');calls=[];seen=[]
        def refill():calls.append(1);return str(path)
        with self.assertRaisesRegex(ValueError,'NEED_IMAGES'):
            self.run_request('A',path,refill,on_token=lambda t,c:seen.append(t))
        self.assertEqual(calls,[]);self.assertEqual(seen,[120])
        self.assertEqual(sum(e.get('event')=='request' for e in self.events()),1)

    def test_precancel_sends_no_image_request_and_no_callback(self):
        cancel=threading.Event();cancel.set();calls=[];path=self.image('A')
        self.assertEqual(self.run_request('A',path,lambda:calls.append(1),cancel),b'')
        self.assertEqual(calls,[]);self.assertFalse(any(e.get('event')=='request' for e in self.events()))

    def test_cancel_during_refill_never_resends_full_file(self):
        cancel=threading.Event();entered=threading.Event();release=threading.Event();calls=[];path=self.image('A')
        def refill():
            calls.append(1);entered.set();self.assertTrue(release.wait(2));path.write_text('FULL:A');return str(path)
        request=self.pool.submit(self.run_request,'A',path,refill,cancel)
        self.assertTrue(entered.wait(2));cancel.set();self.engine.stop_if_running(cancel);release.set()
        self.assertEqual(request.result(timeout=3),b'');self.assertEqual(calls,[1])
        self.assertEqual([e['mode'] for e in self.events() if e.get('event')=='request'],['REF:A'])

    def test_cancel_after_retry_stops_current_slot(self):
        cancel=threading.Event();path=self.image('A');calls=[]
        def refill():calls.append(1);path.write_text('FULL:A');return str(path)
        def on_token(token,event):event.set();self.engine.stop_if_running(event)
        result=self.run_request('A',path,refill,cancel,on_token)
        self.assertTrue(result.startswith(b'r'));self.assertEqual(calls,[1])
        self.assertTrue(any(e['event']=='bstop' for e in self.events()))

    def test_service_reference_refill_owns_fifo_and_emits_whole_body_once(self):
        from serve.frontend import ChatTemplate,openai_to_messages
        from serve.test_image_ids import KeyedVision,png
        tok=S.ByteTokenizer();vision=KeyedVision(self.directory)
        svc=S.Service(self.engine,tok,ChatTemplate(S.ROOT/'serve/chat_template.jinja'),vision=vision)
        cancel=threading.Event();refills=[];original=svc._pictures
        def pictures(*args,**kwargs):
            if kwargs.get('by_reference') is False:
                self.assertTrue(svc.fifo.locked(),'real Service refill did not own FIFO')
                self.assertTrue(self.engine.ctl.acquire(blocking=False),'Service refill retained admission ctl')
                self.engine.ctl.release()
                with self.engine.wlock:self.assertNotIn(cancel,getattr(self.engine,'_slot_cancel_owners',{}))
                refills.append(1)
            return original(*args,**kwargs)
        messages,tools,kwargs=openai_to_messages({'messages':[{'role':'user','content':[
          {'type':'text','text':'look'}, {'type':'image_url','image_url':{'url':png(1)},'uuid':'public-A'}]}],
          'chat_template_kwargs':{'enable_thinking':False}})
        with mock.patch.object(svc,'_pictures',side_effect=pictures):
            ids,thinking,maximum=svc.prepare(messages,tools,kwargs,32,cancel=cancel)
            events=list(svc.run(ids,thinking,tools,maximum,{},cancel))
        text=''.join(ev.text for kind,ev in events if kind=='event' and ev.kind=='content')
        self.assertEqual(text,'reply-A');self.assertEqual(refills,[1]);self.assertEqual(vision.encodes,1)
        self.assertEqual([e['mode'] for e in self.events() if e.get('event')=='request'],['REF:A','FULL:A'])
        self.assertEqual(sum(kind=='done' for kind,ev in events),1)

    def test_refill_exception_releases_claim_and_allows_next_request(self):
        path=self.image('A');cancel=threading.Event()
        def refill():raise RuntimeError('public fake encoder failed')
        with self.assertRaisesRegex(RuntimeError,'encoder failed'):
            self.run_request('A',path,refill,cancel)
        with self.engine.wlock:self.assertNotIn(cancel,getattr(self.engine,'_slot_cancel_owners',{}))
        self.assertEqual(self.run_request('B',None,None),b'reply-B')

    def test_process_identity_change_during_refill_rejects_even_same_generation(self):
        path=self.image('A')
        class ChangedProcess:
            def __init__(self,process):self.process=process
            def __getattr__(self,name):return getattr(self.process,name)
        def refill():
            # A restart publishes its new process before finishing READY metadata
            # and incrementing gen. A new identity must invalidate the old claim.
            self.engine.proc=ChangedProcess(self.engine.proc)
            path.write_text('FULL:A');return str(path)
        with self.assertRaisesRegex(S.EngineDied,'changed'):
            self.run_request('A',path,refill)
        self.assertEqual(sum(e.get('event')=='request' for e in self.events()),1)

    def test_restart_during_refill_does_not_resend_to_new_generation(self):
        path=self.image('A')
        def refill():
            self.engine.gen+=1;path.write_text('FULL:A');return str(path)
        with self.assertRaisesRegex(S.EngineDied,'changed'):
            self.run_request('A',path,refill)
        self.assertEqual(sum(e.get('event')=='request' for e in self.events()),1)


if __name__=='__main__':unittest.main()
