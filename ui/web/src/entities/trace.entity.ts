import { Column, Entity, PrimaryColumn } from 'typeorm';

@Entity()
export class Trace {
  @PrimaryColumn()
  user_id: number;

  @PrimaryColumn()
  action: string;

  @PrimaryColumn()
  created_at: Date;

  @PrimaryColumn({
    type: 'text',
  })
  info: string;
}
