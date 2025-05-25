import { Module } from '@nestjs/common';
import { User } from 'src/entities/user.entity';
import { TypeOrmModule } from '@nestjs/typeorm';

import { UserService } from './user.service';
import { UserController } from './user.controller';
import { IpcChannelModule } from 'src/ipc_channel/ipc_channel.module';

@Module({
  imports: [TypeOrmModule.forFeature([User]), IpcChannelModule],
  controllers: [UserController],
  providers: [UserService],
})
export class UserModule {}
