import { Body, Controller, Get, Post } from '@nestjs/common';

import { UserService } from './user.service';

@Controller('user')
export class UserController {
  constructor(private readonly userService: UserService) {}

  @Post('page')
  async findAll(@Body() { filter, pageNo = 1, pageSize = 10 }: any) {
    const [result, totalCount] = await this.userService.filterByPage({
      filter: filter || {},
      pageNo,
      pageSize,
    });
    return {
      items: result,
      pageNo,
      pageSize,
      totalPage: Math.ceil(totalCount / pageSize),
      totalCount,
    };
  }

  @Post('create')
  async createUserAgents(@Body() configs: any) {
    const result = await this.userService.create_user_agents(configs);
    return { code: 0, success: true, data: '', message: result };
  }
}
