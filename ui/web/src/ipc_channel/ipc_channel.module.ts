import { Module } from '@nestjs/common';
import { IpcChannelService } from './ipc_channel.service';
import { IpcChannelController } from './ipc_channel.controller';

@Module({
  providers: [IpcChannelService],
  exports: [IpcChannelService],
  controllers: [IpcChannelController],
})
export class IpcChannelModule {}
