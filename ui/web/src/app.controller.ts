import { Controller, Get, Param, Sse } from '@nestjs/common';
import { Observable } from 'rxjs';

import { AppService } from './app.service';

@Controller()
export class AppController {
  constructor(private readonly appService: AppService) {}

  @Get('hello')
  getHello(): string {
    return this.appService.getHello();
  }

  @Sse('calc')
  readFromPython(): Observable<string> {
    return this.appService.calculation();
  }

  @Sse('start')
  start(): Observable<string> {
    return this.appService.startScene();
  }

  @Get('test/:method')
  test(@Param('method') method: string): void {
    // this.appService.test('async_add', { a: 5, b: 6 }, () => {
    //   console.log('hi there');
    // });
    this.appService.test(method, { a: 5, b: 6 }, () => {
      console.log('hi there');
    });
  }

  @Get('stop')
  closeSerialPort(): any {
    const result = this.appService.stop();
    // return { success: true, code: 0, message: '', data: result };
    return result;
  }
}
