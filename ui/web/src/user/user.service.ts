import { Injectable, Logger } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { join } from 'node:path';
import { User } from 'src/entities/user.entity';
import { IpcChannelService } from 'src/ipc_channel/ipc_channel.service';
import { Repository } from 'typeorm';
import { promises as fs } from 'fs';

@Injectable()
export class UserService {
  private readonly logger = new Logger(UserService.name);
  constructor(
    @InjectRepository(User) private repository: Repository<User>,
    private ipcRepo: IpcChannelService
  ) {}

  async filterByPage({ filter, pageNo, pageSize }) {
    return this.repository.findAndCount({
      where: filter,
      order: {
        created_at: 'DESC',
      },
      take: pageSize,
      skip: (pageNo - 1) * pageSize,
    });
  }

  async create_user_agents(configs: {
    count: number;
    params: any;
    prompts: string;
  }) {
    return new Promise((resolve, reject) => {
      this.ipcRepo.callPythonFunc(
        'generate_user_data',
        configs,
        async (err, res) => {
          if (err) {
            reject(err);
          }

          const dataDir = join(process.cwd(), `../../data/user_data`); // Save to project_root/data/
          try {
            // Ensure the data directory exists
            await fs.mkdir(dataDir, { recursive: true });

            // Write JSON to file
            const filePath = join(dataDir, `user_data_${configs.count}.json`);
            await fs.writeFile(filePath, JSON.stringify(res, null, 2), 'utf8'); // Pretty-print JSON

            resolve(
              `Generated ${configs.count} user profiles and saved to ${filePath}`
            );
          } catch (error) {
            throw new Error(`Failed to save file: ${error.message}`);
          }
        }
      );
    });
  }
}
