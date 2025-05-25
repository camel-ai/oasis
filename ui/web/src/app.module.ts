import { Module } from '@nestjs/common';
import { TypeOrmModule, TypeOrmModuleOptions } from '@nestjs/typeorm';

import { AppService } from './app.service';
import { AppController } from './app.controller';
import { TraceModule } from './trace/trace.module';
import { UserModule } from './user/user.module';
import { User } from './entities/user.entity';
import { IpcChannelModule } from './ipc_channel/ipc_channel.module';

@Module({
  imports: [
    TypeOrmModule.forRoot({
      type: 'sqlite',
      database: '../../data/reddit_simulation.db',
      entities: [__dirname + '/**/entities/*.entity.js'],
      extra: { connectionLimit: 30 },
      // synchronize: true,
      autoLoadEntities: true,
    } as TypeOrmModuleOptions),
    UserModule,
    TraceModule,
    IpcChannelModule,
  ],
  controllers: [AppController],
  providers: [AppService],
})
export class AppModule {}
