import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { Trace } from 'src/entities/trace.entity';
import { User } from 'src/entities/user.entity';
import { Comment } from 'src/entities/comment.entity';

import { TraceController } from './trace.controller';
import { TraceService } from './trace.service';

@Module({
  imports: [TypeOrmModule.forFeature([Trace, User, Comment])],
  controllers: [TraceController],
  providers: [TraceService],
})
export class TraceModule {}
