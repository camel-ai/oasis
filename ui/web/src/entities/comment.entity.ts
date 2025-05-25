import { Column, Entity, PrimaryColumn } from 'typeorm';

@Entity()
export class Comment {
  @PrimaryColumn()
  comment_id: number;

  @Column()
  user_id: number;

  @Column()
  post_id: number;

  @Column()
  created_at: Date;

  @Column({
    type: 'text',
  })
  content: string;

  @Column()
  num_likes: number;

  @Column()
  num_dislikes: number;
}
