import { Fragment, memo } from 'react';

import UserData from '../../../../../../data/reddit/user_data_36.json';
import GlowBorderUnit from 'app/widgets/glow-border-unit';
import { ArrowLeftIcon, FileTextIcon } from 'lucide-react';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from 'app/components/ui/alert-dialog';

const SimuAgentList = memo(function SimuAgentList({
  back,
}: {
  back: () => void;
}) {
  return (
    <>
      <GlowBorderUnit
        containerClass='bg-white/50 rounded backdrop-blur w-full h-full'
        bodyClass='rounded flex flex-col w-full h-full'
      >
        <div className='flex items-center gap-x-4 p-4 flex-none'>
          <button
            className='bg-white p-2 rounded cursor-pointer'
            onClick={back}
          >
            <ArrowLeftIcon className='size-4' />
          </button>
          Sim Agents List
          <div className='bg-black/10 py-1.5 px-4 rounded ms-auto flex gap-x-2 items-center'>
            <FileTextIcon className='size-4' />
            <span className='text-black/50'>reddit/user_data_36.json</span>
            <AlertDialog>
              <AlertDialogTrigger asChild>
                <button className='cursor-pointer text-[#0055c8] font-bold'>
                  Change
                </button>
              </AlertDialogTrigger>
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>Coming soon...</AlertDialogTitle>
                  <AlertDialogDescription>
                    Will be released by next sprint
                    <br />
                    &nbsp;
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogAction>Confirm</AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          </div>
        </div>
        <div className='flex-1 overflow-y-auto'>
          <table className='w-full border-collapse table-fixed text-black/75 relative bg-white'>
            <thead className='sticky top-0'>
              <tr className=' bg-white'>
                <th className='bg-neutral-700 text-white w-10'></th>
                <th className='bg-neutral-700 text-white w-42'>name</th>
                <th className='bg-neutral-700 text-white w-46'>account</th>
                <th className='bg-neutral-700 text-white w-12'>age</th>
                <th className='bg-neutral-700 text-white w-12'>gender</th>
                <th className='bg-neutral-700 text-white w-16'>MBTI</th>
                <th className='bg-neutral-700 text-white w-24'>nationality</th>
                <th className='py-2 bg-neutral-700 text-white'>profession</th>
              </tr>
            </thead>
            <tbody>
              {UserData.map((user, userIdx) => (
                <Fragment key={`user_${userIdx}`}>
                  <tr>
                    <td className='p-2 text-end border-b border-r border-neutral-200'>
                      {userIdx + 1}
                    </td>
                    <td className='p-2 border-b border-r border-neutral-200'>
                      {user.realname}
                    </td>
                    <td className='p-2 border-b border-r border-neutral-200'>
                      {user.username}
                    </td>
                    <td className='p-2 text-center border-b border-r border-neutral-200'>
                      {user.age}
                    </td>
                    <td className='p-2 text-center border-b border-r border-neutral-200'>
                      {user.gender == 'male' ? '男' : '女'}
                    </td>
                    <td className='p-2 text-center border-b border-r border-neutral-200'>
                      {user.mbti.toUpperCase()}
                    </td>
                    <td className='p-2 border-b border-r border-neutral-200'>
                      {user.country}
                    </td>
                    <td className='p-2 border-b border-neutral-200 line-clamp-1 overflow-hidden'>
                      {user.profession}
                    </td>
                  </tr>
                  <tr>
                    <td colSpan={8} className='p-2 border-b border-neutral-200'>
                      <div className='bg-black/10 rounded p-2'>{user.bio}</div>
                      <span className='text-black/50'>{user.persona}</span>
                    </td>
                  </tr>
                </Fragment>
              ))}
            </tbody>
          </table>
        </div>
      </GlowBorderUnit>
    </>
  );
});

export default SimuAgentList;
