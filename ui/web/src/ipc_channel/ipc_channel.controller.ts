import { Controller, Get, Sse } from '@nestjs/common';
import { IpcChannelService } from './ipc_channel.service';
import { Observable } from 'rxjs';

@Controller('ipc-channel')
export class IpcChannelController {
  constructor(private ipcChannelService: IpcChannelService) {}

  @Sse('start')
  async startTask(): Promise<Observable<any>> {
    return this.ipcChannelService.startToInvoke();
  }

  @Get('stop')
  stopTask() {
    return this.ipcChannelService.stopInvoke();
  }
}
