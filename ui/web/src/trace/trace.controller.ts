import { Body, Controller, Get, Param, Post, Sse } from '@nestjs/common';
import { Observable } from 'rxjs';

import { TraceService } from './trace.service';

@Controller('trace')
export class TraceController {
  constructor(private readonly traceService: TraceService) {}

  @Sse('start')
  async startTask(): Promise<Observable<any>> {
    return this.traceService.startTaskAndFetch();
  }

  @Get('stop')
  stopTask() {
    return this.traceService.stopTask();
  }

  @Post('action/:method')
  taskAction(@Param('method') method: string, @Body() params: any): void {
    this.traceService.callPythonFunc(method, params, () => {
      console.log(`Action [${method}] taken!`);
    });
  }
}
