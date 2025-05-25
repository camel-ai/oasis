import { ChildProcessWithoutNullStreams, spawn } from 'node:child_process';

import { Injectable, OnModuleDestroy } from '@nestjs/common';
import { BehaviorSubject } from 'rxjs';

export const HELLO_MESSAGE = 'Hello React Router 7 from Nest!';

@Injectable()
export class AppService implements OnModuleDestroy {
  private processSub: BehaviorSubject<string>;
  private py: ChildProcessWithoutNullStreams;

  // #requestId = 0;
  // #pendingCallbacks = new Map();
  #buffer = '';

  constructor() {}
  onModuleDestroy() {
    this.py.kill();
    this.processSub.complete();
  }

  getHello(): string {
    return HELLO_MESSAGE;
  }

  stop() {
    this.py.kill();
    this.processSub.complete();
  }

  startScene() {
    // this.#requestId = 0;
    // this.#pendingCallbacks = new Map();
    this.#buffer = '';

    this.processSub = new BehaviorSubject(
      JSON.stringify({ output: 'start to load python code' })
    );

    this.py = spawn(
      'python3',
      ['-u', '../../examples/child_process_bridge/call_from_nodejs.py'],
      {
        stdio: ['pipe', 'pipe', 'inherit'],
      }
    );
    this.py.stdout.on('data', (data) => {
      this.#buffer += data.toString();

      // 处理可能的多条响应
      const responses = this.#buffer.split('\n');
      this.#buffer = responses.pop(); // 剩余部分保留

      responses.forEach((rawResponse) => {
        if (!rawResponse.trim()) return;

        try {
          console.log('raw', rawResponse);
          // const result = rawResponse;

          if (!this.isValidJSON(rawResponse)) {
            this.processSub.next(JSON.stringify({ log: rawResponse }));
            return;
          }
          const { id, result, error } = JSON.parse(rawResponse);
          this.processSub.next(JSON.stringify({ output: { result, error } }));

          // const callback = this.#pendingCallbacks.get(id);

          // if (callback) {
          //   const cb = callback(error, result);
          //   this.#pendingCallbacks.delete(id);

          //   console.log(cb);
          // }
        } catch (e) {
          console.error('解析响应失败:', e);
        }
      });
    });

    return this.processSub;
  }

  isValidJSON(str) {
    try {
      JSON.parse(str);
      return true;
    } catch (e) {
      return false;
    }
  }

  test(method, params, callback) {
    // const reqId = ++this.#requestId;
    const request = JSON.stringify({
      // id: reqId,
      method,
      params,
    });

    console.log(request);
    // this.#pendingCallbacks.set(reqId, callback);
    this.py.stdin.write(request + '\n'); // 添加换行符作为结束
  }

  calculation(path = 'examples/reddit_simulation_params.py') {
    this.processSub = new BehaviorSubject(
      JSON.stringify({ output: 'start to load python code' })
    );
    if (this.py) this.py.kill();
    this.py = spawn('python3', ['../../' + path, '../../']);

    this.py.stdout.on('data', (data) => {
      this.processSub.next(JSON.stringify({ output: data.toString() }));
    });
    this.py.stderr.on('data', (data) => {
      this.processSub.next(JSON.stringify({ error: data.toString() }));
    });

    this.py.on('close', (code) => {
      this.processSub.next(JSON.stringify({ status: code }));
      this.processSub.unsubscribe();
    });

    return this.processSub;
  }
}
