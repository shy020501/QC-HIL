# utils/async_saver.py
import os, time, threading, pickle, queue
import flax

class AsyncSaver:
    """
    비동기 체크포인트 저장기.
    - 메인 스레드는 request()로 저장 요청만 넣고 즉시 리턴
    - 워커 스레드가 최신 요청만 집어서 디스크에 원자적으로 저장
    """
    def __init__(self, save_dir, params_lock, max_pending=2):
        self.save_dir = save_dir
        self.params_lock = params_lock
        self.q = queue.Queue(maxsize=max_pending)
        self._alive = True
        self._worker = threading.Thread(target=self._loop, name="async_saver", daemon=True)
        self._worker.start()

    def request(self, agent, step, params_ref=None):
        """저장 요청. 큐가 가득 차면 오래된 요청 버리고 최신만 유지."""
        item = (time.time(), int(step), agent, params_ref)
        try:
            self.q.put_nowait(item)
        except queue.Full:
            try:
                self.q.get_nowait()
                self.q.task_done()
            except queue.Empty:
                pass
            try:
                self.q.put_nowait(item)
            except queue.Full:
                pass

    def _snapshot(self, agent, step, params_ref):
        with self.params_lock:
            if params_ref is not None and 'params' in params_ref:
                net = agent.network.replace(params=params_ref['params'])
                agent_to_save = agent.replace(network=net)
            else:
                agent_to_save = agent

        save_dict = dict(agent=flax.serialization.to_state_dict(agent_to_save))
        tmp = os.path.join(self.save_dir, f'params_{step}.pkl.tmp')
        final = os.path.join(self.save_dir, f'params_{step}.pkl')
        with open(tmp, 'wb') as f:
            pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, final)

    def _loop(self):
        last_done_step = None
        while self._alive:
            try:
                _, step, agent, params_ref = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if last_done_step != step:
                    self._snapshot(agent, step, params_ref)
                    last_done_step = step
            finally:
                self.q.task_done()

    def flush_and_stop(self, agent, step, params_ref=None, timeout=3.0):
        self.request(agent, step, params_ref)
        end = time.time() + timeout
        while not self.q.empty() and time.time() < end:
            time.sleep(0.05)
        self._alive = False
