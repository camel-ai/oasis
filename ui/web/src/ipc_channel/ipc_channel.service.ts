import {
  Injectable,
  InternalServerErrorException,
  Logger,
  OnModuleDestroy,
} from '@nestjs/common';
import { ChildProcessWithoutNullStreams, spawn } from 'node:child_process';
import { BehaviorSubject, Subscription } from 'rxjs';

@Injectable()
export class IpcChannelService implements OnModuleDestroy {
  private readonly logger = new Logger(IpcChannelService.name);
  private processSub: BehaviorSubject<string>;
  private py: ChildProcessWithoutNullStreams;

  #requestId = 0;
  #pendingCallbacks = new Map();
  #buffer = '';

  constructor() {}
  onModuleDestroy() {
    if (this.processSub) this.processSub.complete();
    if (this.py) this.py.kill();

    this.logger.log('Ipc Channel initialized');
  }

  /**
   * stop this invoke and release all the events
   */
  stopInvoke() {
    if (this.processSub) this.processSub.complete();
    if (this.py) this.py.kill();
    this.logger.log('Ipc Channel invoke stopped');
  }

  /**
   * start to invoke python file by :path and ready to output messages
   * @param path python path
   * @returns observable messages
   */
  startToInvoke(
    path = '../../examples/child_process_bridge/call_from_nodejs.py'
  ) {
    this.#requestId = 0;
    this.#pendingCallbacks = new Map();
    this.#buffer = '';

    this.processSub = new BehaviorSubject(
      JSON.stringify({ log: 'start to load python code' })
    );

    this.py = spawn('python3', [path], {
      // this.py = spawn('python3', ['-u', path], {
      //   stdio: ['pipe', 'pipe', 'inherit'],
    });
    this.py.stdout.on('data', (data) => {
      console.debug(data.toString());
      this.#buffer += data.toString();

      // 处理可能的多条响应
      const responses = this.#buffer.split('\n');
      this.#buffer = responses.pop(); // 剩余部分保留

      responses.forEach((rawResponse) => {
        if (!rawResponse.trim()) return;

        try {
          if (!this.isValidJSON(rawResponse)) {
            this.processSub.next(JSON.stringify({ log: rawResponse }));
            return;
          }
          const { id, result, error } = JSON.parse(rawResponse);
          this.processSub.next(JSON.stringify({ output: { result, error } }));

          const callback = this.#pendingCallbacks.get(id);
          if (callback) {
            const cb = callback(error, result);
            this.#pendingCallbacks.delete(id);
          }
        } catch (e) {
          this.logger.error(`Error parsing messages from python file: ${e}`);
        }
      });
    });
    // 错误处理
    this.py.on('error', (err) => {
      this.logger.error(`Child process error: ${err}`);
    });

    this.py.on('close', (code) => {
      this.logger.log(`Child process exit with code: ${code}`);
      // 清理未完成的回调
      this.#pendingCallbacks.forEach((cb) =>
        cb(new Error('Child process terminated.'))
      );
      this.#pendingCallbacks.clear();
    });

    this.logger.log('Ipc Channel invoke started');
    return this.processSub;
  }

  /**
   * call a function in python file
   * @param method python function name
   * @param params params for function
   * @param callback return in (error, result) format
   */
  callPythonFunc(method, params?, callback?) {
    if (!this.py || !this.processSub)
      throw new InternalServerErrorException(
        'Python Ipc Channel has not been initialized'
      );

    const reqId = ++this.#requestId;
    const request = JSON.stringify({
      id: reqId,
      method,
      params,
    });

    this.#pendingCallbacks.set(reqId, callback);
    this.py.stdin.write(request + '\n'); // 添加换行符作为结束
    this.logger.log(
      // `Ipc Channel call for function ${method} with params: ${JSON.stringify(params)}`
      `Ipc Channel call for function ${method}`
    );
  }

  private isValidJSON(str: string) {
    try {
      JSON.parse(str);
      return true;
    } catch (e) {
      return false;
    }
  }
}
