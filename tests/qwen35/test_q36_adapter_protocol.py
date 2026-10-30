#!/usr/bin/env python3
"""CPU regression of real q36_parallel.hpp using explicit fake native/model stub.
No evidence of GPU math, KV correctness, grammar or actual native overlap.
"""
import os, queue, subprocess, threading, time, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent
MOCK_BINARY=None
_BUILD_TEMP=None
def setUpModule():
    global MOCK_BINARY,_BUILD_TEMP
    supplied=os.environ.get("Q36_ADAPTER_MOCK_BIN")
    if supplied:
        MOCK_BINARY=Path(supplied).resolve()
    else:
        import tempfile,shlex
        _BUILD_TEMP=tempfile.TemporaryDirectory()
        MOCK_BINARY=Path(_BUILD_TEMP.name)/"q36_adapter_protocol_mock"
        compiler=shlex.split(os.environ.get("CXX","c++"))
        subprocess.run([*compiler,"-std=c++17","-O0","-pthread",str(ROOT/"q36_adapter_protocol_mock.cpp"),"-o",str(MOCK_BINARY)],check=True)
def tearDownModule():
    if _BUILD_TEMP is not None:_BUILD_TEMP.cleanup()
class Session:
    def __init__(self,extra=None):
        self.started=time.monotonic()
        self.p=subprocess.Popen([str(MOCK_BINARY)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1,env={**os.environ,'STRATA_Q36_BATCH_TRACE':'1',**(extra or {})})
        self.q=queue.Queue();self.lines=[];self.err=[]
        def read(pipe,q=None):
            for line in pipe:
                line=line.strip()
                if q:q.put(line)
                else:self.err.append(line)
        self.reader=threading.Thread(target=read,args=(self.p.stdout,self.q),daemon=True);self.reader.start()
        self.er=threading.Thread(target=read,args=(self.p.stderr,),daemon=True);self.er.start();self.until(lambda l:l.startswith('READY '));self.ready_elapsed=time.monotonic()-self.started
    def send(self,line):self.p.stdin.write(line+'\n');self.p.stdin.flush()
    def until(self,pred,timeout=3):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            try:l=self.q.get(timeout=max(.001,deadline-time.monotonic()))
            except queue.Empty:break
            self.lines.append(l)
            if pred(l):return l
        raise AssertionError('timeout; transcript='+repr(self.lines))
    def close(self,eof=False):
        if self.p.poll() is None:
            if eof:self.p.stdin.close()
            else:self.send('QUIT')
            try:self.p.wait(timeout=3)
            except subprocess.TimeoutExpired:self.p.kill();self.p.wait();raise AssertionError('adapter did not join workers on EOF/QUIT')
        self.reader.join(timeout=1);self.er.join(timeout=1)
        for f in (self.p.stdin,self.p.stdout,self.p.stderr):
            if not f.closed:f.close()
        return self.p.returncode
class Adapter(unittest.TestCase):
    def setUp(self):self.s=Session();self.addCleanup(self.s.close)
    def test_isolated_slots_have_busy_and_cancel_other_progress(self):
        s=self.s;s.send('BGEN 0 40 1,2,3');s.until(lambda l:l=='BADM 0 1')
        s.send('BGEN 1 40 4,5,6');s.until(lambda l:l=='BADM 1 1')
        s.send('HAVE 2a,ff');self.assertEqual(s.until(lambda l:l.startswith('HAVE')),'HAVE 3:1:1,0')
        s.send('BGEN 1 1 9');self.assertEqual(s.until(lambda l:l.startswith('ERR')),'ERR BGEN: no such free slot')
        s.send('BSTOP 0');done0=s.until(lambda l:l.startswith('BDONE 0 '));self.assertIn(' cancel ',done0)
        done1=s.until(lambda l:l.startswith('BDONE 1 '));self.assertIn(' length ',done1)
        self.assertIn(' 40 length ',done1)
        admissions=[l for l in s.lines if l.startswith('DONE 1 ')]
        self.assertTrue(all(l=='DONE 1 10 1.500000 0 length 0 0 7' for l in admissions),admissions)
        for i,l in enumerate(s.lines):
            if l.startswith('T '):
                self.assertTrue(s.lines[i+1].startswith('DONE 1 '))
                self.assertTrue(s.lines[i+2].startswith('BADM '))
        self.assertTrue(all(l=='BT 0 100' for l in s.lines if l.startswith('BT 0 ')))
        self.assertTrue(all(l=='BT 1 101' for l in s.lines if l.startswith('BT 1 ')))
        self.assertTrue(any(l=='BT 1 101' for l in s.lines[s.lines.index(done0)+1:]))
        s.send('BGEN 0 2 1');s.until(lambda l:l=='BADM 0 1');s.until(lambda l:l.startswith('BDONE 0 '))
        self.assertEqual(s.close(),0)
    def test_eof_joins_active_workers(self):
        s=self.s;s.send('BGEN 0 1000 1');s.until(lambda l:l=='BADM 0 1');s.send('BGEN 1 1000 2');s.until(lambda l:l=='BADM 1 1')
        self.assertEqual(s.close(eof=True),0)
    def test_solo_gen_and_invalid_slot(self):
        s=self.s;s.send('BGEN 99 1 1');self.assertEqual(s.until(lambda l:l.startswith('ERR')),'ERR BGEN: invalid slot')
        s.send('GEN 2 1,2');s.until(lambda l:l.startswith('DONE '));self.assertFalse(any(l.startswith(('BT ','BADM ','BDONE ')) for l in s.lines))
    def test_error_does_not_release_slot_before_idle_cleanup(self):
        s=self.s;s.send('BGEN 0 20 777');s.until(lambda l:l.startswith('MOCK cleaning'))
        s.send('BGEN 0 20 1');reply=s.until(lambda l:l.startswith(('ERR BGEN:','T ')))
        self.assertEqual(reply,'ERR BGEN: no such free slot','error released busy before native cleanup callback')
    def test_deferred_final_keeps_slot_busy_until_idle(self):
        s=self.s;s.send('BGEN 0 2 1');s.until(lambda l:l=='BADM 0 1');s.until(lambda l:l=='BT 0 100')
        s.send('BGEN 0 1 1');self.assertEqual(s.until(lambda l:l.startswith('ERR')),'ERR BGEN: no such free slot')
        s.until(lambda l:l.startswith('BDONE 0 '));s.send('BGEN 0 1 1');s.until(lambda l:l=='BADM 0 1');s.until(lambda l:l.startswith('BDONE 0 '))
    def test_cancel_queued_worker_waiting_control_gate(self):
        s=Session({'MOCK_SLOW_ADMIT':'1'});self.addCleanup(s.close)
        s.send('BGEN 0 20 1');s.until(lambda l:l=='MOCK admission')
        s.send('BGEN 1 20 1');time.sleep(.02);s.send('BSTOP 1')
        s.until(lambda l:l=='BADM 0 1')
        done=s.until(lambda l:l.startswith('DONE 0 '));self.assertIn(' cancel ',done)
        self.assertEqual(s.until(lambda l:l.startswith('BADM 1 ')),'BADM 1 0')
        s.until(lambda l:l.startswith('BDONE 0 '))
        self.assertFalse(any(l.startswith('BT 1 ') for l in s.lines))
    def test_post_admission_error_scoped_without_control_reply_theft(self):
        s=self.s;s.send('BGEN 0 40 888');s.until(lambda l:l=='BADM 0 1')
        s.send('BGEN 1 40 2');s.until(lambda l:l=='BADM 1 1')
        scoped=s.until(lambda l:l.startswith('BDONE 0 '));self.assertIn(' error ',scoped)
        self.assertFalse(any(l.startswith('ERR mock postadmit') for l in s.lines))
        other=s.until(lambda l:l.startswith('BDONE 1 '));self.assertIn(' 40 length ',other)
    def test_reclaim_retained_maps_from_queued_but_not_gpu_active_owner(self):
        s=Session({'MOCK_RECLAIM':'1','MOCK_SLOW_ADMIT_SLOT1':'1'});self.addCleanup(s.close)
        s.send('BGEN 0 2 1');s.until(lambda l:l.startswith('BDONE 0 '))
        # Slot0 retains1.6MB while cap2.4MB cannot hold two requests.
        s.send('BGEN 1 20 1');s.until(lambda l:l=='MOCK admission')
        # Slot0 now busy waiting for control, but its old maps are safely idle.
        s.send('BGEN 0 2 1')
        s.until(lambda l:l=='BADM 1 1');s.until(lambda l:l.startswith('BDONE 1 '))
        s.until(lambda l:l=='BADM 0 1');s.until(lambda l:l.startswith('BDONE 0 '))
    def test_image_reference_width_and_owned_cache_capabilities(self):
        info=next(l for l in self.s.lines if l.startswith('INFO engine=q36'))
        self.assertIn('img_refs=1',info);self.assertIn('n_embd=2048',info)
        self.assertIn('batch_pinned=1',info);self.assertIn('slot_cache=0',info)
        for env,wanted in [({'MOCK_HOST_PARK':'1'},'slot_cache=1'),({'MOCK_HOST_PARK':'1','MOCK_NO_SNAPS':'1'},'slot_cache=0')]:
            s=Session(env);self.addCleanup(s.close)
            self.assertIn(wanted,next(l for l in s.lines if l.startswith('INFO engine=q36')))
    def test_prefill_chunk_metadata_and_actual_owner_init(self):
        self.assertIn('batch_prefill_chunk=64',next(l for l in self.s.lines if l.startswith('INFO engine=q36')))
        s=Session({'MOCK_PREFILL_CHUNK':'256','MOCK_REQUIRE_CHUNK':'256'});self.addCleanup(s.close)
        self.assertIn('batch_prefill_chunk=256',next(l for l in s.lines if l.startswith('INFO engine=q36')))
    def test_shared_host_cache_pool_is_initialized_once_before_workers(self):
        s=Session({'MOCK_HOST_PARK':'1'});self.addCleanup(s.close)
        info=next(l for l in s.lines if l.startswith('INFO engine=q36'))
        self.assertIn('host_cache_shared=1',info);self.assertIn('snapshots=1',info);self.assertIn('host_park_mib=1',info)
        self.assertEqual(s.close(),0)
        self.assertIn('MOCK_HUB_ALLOCS snapshots=1 pages=1',s.err)
    def test_explicit_single_owner_parallel_api_uses_shared_metadata(self):
        # Native CLI defaultbatch1 intentionally stays legacy; this tests the
        # shared executor API with one owner, not a claim about CLI dispatch.
        s=Session({'MOCK_SLOTS':'1','MOCK_HOST_PARK':'1'});self.addCleanup(s.close)
        info=next(l for l in s.lines if l.startswith('INFO engine=q36'))
        self.assertIn('batch_slots=1',info);self.assertIn('host_cache_shared=1',info)
        s.send('BGEN 0 2 1');s.until(lambda l:l=='BADM 0 1');s.until(lambda l:l.startswith('BDONE 0 '))
        self.assertEqual(s.close(),0);self.assertIn('MOCK_HUB_ALLOCS snapshots=1 pages=1',s.err)
    def test_global_ready_waits_for_last_native_worker_ready(self):
        s=Session({'MOCK_DELAY_READY':'1'});self.addCleanup(s.close)
        self.assertGreaterEqual(s.ready_elapsed,.13)
    def test_mapping_failure_refuses_without_wait_or_fake_success(self):
        s=Session({'MOCK_MAP_FAIL':'1'});self.addCleanup(s.close)
        s.send('BGEN 0 4 1');self.assertEqual(s.until(lambda l:l.startswith('ERR')),'ERR mock admission refused')
        self.assertFalse(any(l.startswith(('T ','BADM 0 1')) for l in s.lines))
    def test_cancel_admission_waiting_for_shared_budget(self):
        s=Session({'MOCK_PRESSURE':'1'});self.addCleanup(s.close)
        s.send('BGEN 0 4 1');s.until(lambda l:l=='MOCK admission');s.send('BSTOP 0')
        self.assertIn(' cancel ',s.until(lambda l:l.startswith('DONE 0 ')))
        self.assertEqual(s.until(lambda l:l.startswith('BADM ')),'BADM 0 0')
    def test_fatal_executor_failure_ends_process_not_waiters_forever(self):
        s=Session({'MOCK_EXECUTOR_THROW':'1'});self.addCleanup(s.close)
        s.send('BGEN 0 4 1');self.assertNotEqual(s.p.wait(timeout=3),0)
    def test_physical_preflight_refuses_ready(self):
        p=subprocess.run([str(MOCK_BINARY)],input='',text=True,capture_output=True,timeout=3,env={**os.environ,'MOCK_LOW_VRAM':'1'})
        self.assertNotEqual(p.returncode,0);self.assertNotIn('READY ',p.stdout)
class ResidentCheckpoint(unittest.TestCase):
    def test_opt_in_idle_pause_resume_and_busy_refusal(self):
        s=Session({'MOCK_RESIDENT_PREFIX':'1'});self.addCleanup(s.close)
        self.assertTrue(any('resident_prefix_reuse=1 checkpoint_clock=1' in line for line in s.lines))
        s.send('BGEN 0 40 1');s.until(lambda line:line=='BADM 0 1')
        s.send('CHECKPOINT_PAUSE');self.assertEqual(s.until(lambda line:line.startswith('ERR checkpoint:')),'ERR checkpoint: active request')
        s.send('BSTOP 0');s.until(lambda line:line.startswith('BDONE 0 '))
        s.send('CHECKPOINT_PAUSE');self.assertEqual(s.until(lambda line:line.startswith('CHECKPOINT_PAUSED')),'CHECKPOINT_PAUSED graph_execs=0')
        s.send('BGEN 0 1 1');self.assertEqual(s.until(lambda line:line.startswith('ERR checkpoint:')),'ERR checkpoint: paused')
        s.send('CHECKPOINT_RESUME');ack=s.until(lambda line:line.startswith('CHECKPOINT_RESUMED'))
        fields=dict(item.split('=',1) for item in ack.split()[1:])
        self.assertGreaterEqual(int(fields['paused_us']),0);self.assertEqual(fields['graph_execs'],'0')
        s.send('BGEN 0 1 1');s.until(lambda line:line.startswith('DONE '));s.until(lambda line:line.startswith('BADM 0 '))
    def test_default_does_not_advertise_or_handle_checkpoint(self):
        s=Session();self.addCleanup(s.close)
        self.assertTrue(all('resident_prefix_reuse' not in line and 'checkpoint_clock' not in line for line in s.lines))
        s.send('CHECKPOINT_PAUSE');self.assertEqual(s.until(lambda line:line.startswith('ERR ')),'ERR expected GEN/BGEN/HAVE/BSTOP')

if __name__=='__main__':unittest.main(verbosity=2)
