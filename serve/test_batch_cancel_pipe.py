import ast,contextlib,json,os,threading,time,types,unittest
from pathlib import Path
from unittest.mock import patch
P=Path(__file__).with_name('server.py');tree=ast.parse(P.read_text());cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='StrataEngine');token=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='_BatchCancelToken');names={'_send_owned_batch_legacy','_forget_batch_owner_legacy','_stop_if_running_legacy','_close_cancel_pipe','_bstop_command','_try_cancel_pipe','stop_if_running','_send_owned_batch','_bind_active_batch_owner','_forget_batch_owner'};methods=[x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name in names]
class EngineDied(RuntimeError):pass
ns=dict(os=os,time=time,json=json,contextlib=contextlib,EngineDied=EngineDied);exec(compile(ast.fix_missing_locations(ast.Module(body=[token,ast.ClassDef(name='Candidate',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[])),str(P),'exec'),ns)
class Writer:
 def __init__(self):self.lines=[]
 def write(self,x):self.lines.append(x)
 def flush(self):pass
class Tests(unittest.TestCase):
 def engine(self):
  e=ns['Candidate']();e._batch_cancel_enabled=True;e.wlock=threading.Lock();e.cancel_owner_lock=threading.RLock();e.gen=4;e.proc=types.SimpleNamespace(stdin=Writer());e.info={'batch_cancel_epoch':'1','batch_cancel_fd':'1','batch_pinned':'0'};e.can_stop=True;e.alive=lambda:True;e._running_cancel=None;r,w=os.pipe();os.set_blocking(w,False);os.set_blocking(r,False);e._cancel_pipe_write=w;self.addCleanup(lambda:os.close(r));self.addCleanup(e._close_cancel_pipe);return e,r
 def owner(self,e,slot=1):
  c=threading.Event();o=e._send_owned_batch('BGEN 1 32 1,2',c,slot,4);return c,o
 def test_pipe_bypasses_held_stdin_writer_lock(self):
  e,r=self.engine();c,o=self.owner(e);c.set();e.wlock.acquire();done=threading.Event();t=threading.Thread(target=lambda:(e.stop_if_running(c),done.set()));t.start()
  try:self.assertTrue(done.wait(1),'cancel waited on giant stdin writer lock');self.assertEqual(os.read(r,128),b'BSTOP 1 1\n')
  finally:e.wlock.release();t.join()
 def test_admission_owner_is_published_before_BADM(self):
  e,r=self.engine();c,o=self.owner(e);c.set();e.stop_if_running(c);self.assertEqual(os.read(r,128),b'BSTOP 1 1\n');self.assertIn('cancel_epoch=1',e.proc.stdin.lines[0])
 def test_stale_owner_cannot_cancel_new_slot(self):
  e,r=self.engine();old,a=self.owner(e);new,b=self.owner(e);old.set();e.stop_if_running(old)
  with self.assertRaises(BlockingIOError):os.read(r,128)
  e._forget_batch_owner(old,a);new.set();e.stop_if_running(new);self.assertEqual(os.read(r,128),b'BSTOP 1 2\n');self.assertEqual(e._bstop_command(1,a),'BSTOP 1 1')
 def test_generation_or_proc_change_rejects_old_owner(self):
  e,r=self.engine();c,o=self.owner(e);e.gen=5;c.set();e.stop_if_running(c)
  with self.assertRaises(BlockingIOError):os.read(r,128)
 def test_full_pipe_falls_back_to_exact_epoch_stdin(self):
  e,r=self.engine();c,o=self.owner(e);c.set()
  with patch.object(os,'write',side_effect=BlockingIOError):e.stop_if_running(c)
  self.assertEqual(e.proc.stdin.lines[-1],'BSTOP 1 1\n')
 def test_eintr_is_retried_and_fd_close_idempotent(self):
  e,r=self.engine();c,o=self.owner(e);c.set();write=os.write;calls=[0]
  def retry(fd,data):
   calls[0]+=1
   if calls[0]==1:raise InterruptedError()
   return write(fd,data)
  with patch.object(os,'write',retry):e.stop_if_running(c)
  self.assertEqual(os.read(r,128),b'BSTOP 1 1\n');e._close_cancel_pipe();e._close_cancel_pipe();self.assertIsNone(e._cancel_pipe_write)
 def test_legacy_capability_keeps_legacy_protocol(self):
  e,r=self.engine();e.info={'batch_pinned':'0'};c,o=self.owner(e);self.assertNotIn('cancel_epoch=',e.proc.stdin.lines[0]);self.assertNotIn(c,getattr(e,'_slot_cancel_owners',{}));e._bind_active_batch_owner(c,o);c.set();e.stop_if_running(c);self.assertEqual(e.proc.stdin.lines[-1],'BSTOP 1\n')
 def test_partial_frame_disables_channel_before_next_owner(self):
  e,r=self.engine();c,o=self.owner(e);c.set()
  with patch.object(os,'write',return_value=1):e.stop_if_running(c)
  self.assertIsNone(e._cancel_pipe_write);self.assertEqual(e.proc.stdin.lines[-1],'BSTOP 1 1\n')
 def test_pre_cancelled_admission_is_not_written(self):
  e,r=self.engine();c=threading.Event();c.set();self.assertIsNone(e._send_owned_batch('BGEN 1 32 1,2',c,1,4));self.assertEqual(e.proc.stdin.lines,[])
 def test_lazy_or_queued_cancel_never_requires_stdin_lock(self):
  e,r=self.engine();c=threading.Event();c.set();del e.wlock;e.stop_if_running(c);self.assertEqual(e.proc.stdin.lines,[])
 def test_queued_cancel_bypasses_another_owners_held_writer_lock(self):
  e,r=self.engine();c=threading.Event();c.set();done=threading.Event();e.wlock.acquire();t=threading.Thread(target=lambda:(e.stop_if_running(c),done.set()));t.start()
  try:self.assertTrue(done.wait(1),'queued cancellation waited behind unrelated BGEN write')
  finally:e.wlock.release();t.join()
 def test_monitor_accounting_follows_critical_cancel(self):
  source=P.read_text();a=source.index('newly_signalled = not cancel.is_set()');b=source.index('svc.note_client_disconnect(cancel, signalled=newly_signalled)',a);part=source[a:b];self.assertLess(part.index('cancel.set()'),part.index('stopper(cancel)'));self.assertLess(part.index('stopper(cancel)'),part.index('self._note('))
 def test_off_generic_preserves_legacy_control_route(self):
  e,r=self.engine();e._batch_cancel_enabled=False;c,o=self.owner(e)
  self.assertEqual(e.proc.stdin.lines,['BGEN 1 32 1,2\n'])
  e._bind_active_batch_owner(c,o)
  self.assertNotIn(c,getattr(e,'_slot_cancel_owners',{}))
  c.set();e.stop_if_running(c);self.assertEqual(len(e.proc.stdin.lines),1)
  e._running_cancel=c;e.stop_if_running(c);self.assertEqual(e.proc.stdin.lines[-1],'STOP\n')
 def test_off_pinned_preserves_legacy_stop(self):
  e,r=self.engine();e._batch_cancel_enabled=False;e.info['batch_pinned']='1';c,o=self.owner(e)
  c.set();e.stop_if_running(c);self.assertEqual(e.proc.stdin.lines[-1],'BSTOP 1\n')
if __name__=='__main__':unittest.main(verbosity=2)
