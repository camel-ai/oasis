import { Column, Entity, PrimaryColumn } from 'typeorm';

@Entity()
export class User {
  @PrimaryColumn()
  user_id: number;

  @Column()
  agent_id: number;

  @Column()
  user_name: string;

  @Column()
  name: string;

  @Column({
    type: 'text',
  })
  bio: string;

  @Column()
  created_at: Date;

  @Column()
  num_followings: number;

  @Column()
  num_followers: number;
}
