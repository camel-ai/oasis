import { ChildProcessWithoutNullStreams, spawn } from 'node:child_process';

import { Injectable, Logger, OnModuleDestroy } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { BehaviorSubject, interval, Subscription, switchMap } from 'rxjs';
import { Trace } from 'src/entities/trace.entity';
import { Comment } from 'src/entities/comment.entity';
import { DataSource, Repository } from 'typeorm';

@Injectable()
export class TraceService implements OnModuleDestroy {
  private readonly logger = new Logger(TraceService.name);
  private processSub: BehaviorSubject<string>;
  private py: ChildProcessWithoutNullStreams;
  private dataRecordSub: Subscription;

  // #requestId = 0;
  // #pendingCallbacks = new Map();
  #buffer = '';

  constructor(@InjectRepository(Trace) private repositoy: Repository<Trace>) {}
  onModuleDestroy() {
    this.py.kill();
    this.processSub.complete();
    if (this.dataRecordSub) {
      this.dataRecordSub.unsubscribe();
    }
  }

  stopTask() {
    if (this.processSub) this.processSub.complete();
    if (this.py) this.py.kill();
    if (this.dataRecordSub) this.dataRecordSub.unsubscribe();
  }

  startTaskAndFetch() {
    // this.#requestId = 0;
    // this.#pendingCallbacks = new Map();
    this.#buffer = '';

    this.processSub = new BehaviorSubject(
      JSON.stringify({ output: 'start to load python code' })
    );

    this.py = spawn(
      'python3',
      ['-u', '../../examples/reddit_simulation_invoke.py'],
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
          // const { id, result, error } = JSON.parse(rawResponse);
          const result = rawResponse;
          this.processSub.next(JSON.stringify({ output: result }));

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

    // 同步获取 数据库 列表数据
    setTimeout(() => {
      this.fetchDataRecords();
    }, 10000);
    return this.processSub;
  }

  callPythonFunc(method, params, callback) {
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

  fetchDataRecords() {
    let rank = 0;

    const datasource = new DataSource({
      type: 'sqlite',
      database: '../../data/reddit_simulation.db',
      entities: [__dirname + '/../**/entities/*.entity.{js,ts}'],
      synchronize: false,
    });

    this.dataRecordSub = interval(2000)
      .pipe(switchMap(async () => await this.findAfterByTime(datasource, rank)))
      .subscribe((result) => {
        if (result.length !== 0) {
          rank += result.length;
          console.log('taken count: ', result.length);
          this.processSub.next(JSON.stringify({ data: result }));
        }
      });
  }

  async findAfterByTime(dbSource: DataSource, rank: number) {
    console.log('judge rank', rank);

    try {
      await dbSource.initialize();
      const result = await dbSource
        .createQueryBuilder(Trace, 'trace')
        .skip(rank)
        .orderBy({
          'trace.created_at': 'ASC',
        })
        .getMany();
      const commentsTrace = result.filter(
        (res) => res.action === 'create_comment'
      );
      const commentIds = commentsTrace.map(
        (res) => JSON.parse(res.info)['comment_id']
      );
      const comments = await dbSource
        .createQueryBuilder(Comment, 'comment')
        .where('comment.comment_id IN (:...commentIdArr)', {
          commentIdArr: commentIds,
        })
        .getMany();

      result.forEach((res) => {
        if (res.action == 'create_comment') {
          const _commentId = JSON.parse(res.info)['comment_id'];
          const _comment = comments.find((c) => c.comment_id == _commentId);
          res.info = JSON.stringify(_comment);
        }
      });

      return result;
    } catch (err) {
      console.log(err);
      this.logger.error(`动态加载数据库查询发生错误，${err}`);
    } finally {
      dbSource.destroy();
    }
  }
}
