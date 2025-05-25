import clsx from 'clsx';
import {
  BanIcon,
  MessageSquareText,
  PlayIcon,
  SquareIcon,
  SquarePenIcon,
  ThumbsDownIcon,
  ThumbsUpIcon,
  UserPlus2Icon,
} from 'lucide-react';
import { memo, useEffect, useMemo, useRef, useState } from 'react';

import { SSE } from 'sse.js';

import { generateGraph } from 'app/utils';

import { useSimulationContext } from './simulate-context';
import SimuGraph from './graph';
import GlowBorderUnit from 'app/widgets/glow-border-unit';
import SimuConsole from './console';
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

const Task_Action = async (action_name: string) => {
  try {
    const response = await fetch(`/api/trace/action/${action_name}`, {
      method: 'POST',
    });
    const result = response.json();
    console.log('action result: ', result);
  } catch (err) {
    console.log(err);
  }
};
const Stop_Simulation = async () => {
  try {
    const response = await fetch(`/api/trace/stop`);
    const result = response.json();
    console.log('stopped');
  } catch (err) {
    console.log(err);
  }
};

const USER_ACTION = [
  {
    icon: <UserPlus2Icon className='size-4' />,
    title: '关注',
  },
  {
    icon: <ThumbsUpIcon className='size-4' />,
    title: '点赞',
    onClick: () => {
      Task_Action('take_action');
    },
  },
  {
    icon: <ThumbsDownIcon className='size-4' />,
    title: '踩一下',
  },
  {
    icon: <SquarePenIcon className='size-4' />,
    title: '发帖',
  },
  {
    icon: <MessageSquareText className='size-4' />,
    title: '评论',
  },
  {
    icon: <BanIcon className='size-4' />,
    title: '屏蔽',
  },
];

const SimuMonitor = memo(function SimuMonitor({ active }: { active: boolean }) {
  const SimulationContext = useSimulationContext();
  const PageAction = useMemo(() => {
    return {
      running: SimulationContext.running,
      setRunning: SimulationContext.setRunning,
      setGraphData: SimulationContext.setGraphData,
      setConsoleData: SimulationContext.setConsoleData,
    };
  }, [SimulationContext]);

  const sourceRef = useRef<SSE>(null);
  useEffect(() => {
    sourceRef.current = new SSE('/api/trace/start', {
      start: false,
    });
    sourceRef.current.onmessage = ({ data }: any) => {
      // Assuming we receive JSON-encoded data payloads:
      const result = JSON.parse(data);
      // console.log(result);

      if (result.hasOwnProperty('data')) {
        PageAction.setGraphData((data: any) => {
          return generateGraph(result['data'], data);
        });
      }
      if (result.hasOwnProperty('output')) {
        PageAction.setConsoleData(result['output']);
      }
    };
    // eventSource.onerror = (err) => {
    //   console.log('err', err);
    // };
    // eventSource.onopen = (open) => {
    //   console.log('open', open);
    // };
    return () => {
      if (sourceRef.current?.OPEN) {
        sourceRef.current?.close();
      }
      sourceRef.current = null;
    };
  }, []);

  const start = () => {
    PageAction.setRunning(true);
    sourceRef.current?.stream();

    setTimeout(() => {
      Task_Action('initial_scene');
    }, 6000);
    setShowConsole(true);
  };
  const stop = () => {
    sourceRef.current?.close();
    Stop_Simulation();
    PageAction.setRunning(false);
  };

  const [round, setRound] = useState(1);
  const [roundAgentNum, setRoundAgentNum] = useState(10);
  const [showConsole, setShowConsole] = useState(false);

  return (
    <>
      <div
        className={clsx(
          'absolute inset-0 p-2 px-4 h-full flex justify-center items-center transition-transform z-0',
          {
            'translate-x-0': active,
            'translate-x-[100%]': !active,
          }
        )}
      >
        <div className='absolute z-10 top-2 left-1/2 translate-x-[-50%]'>
          <GlowBorderUnit
            containerClass='bg-white/50 rounded-full backdrop-blur'
            bodyClass='!p-2 rounded-full flex gap-x-4 items-center'
          >
            <AlertDialog>
              <AlertDialogTrigger asChild>
                <button
                  className='rounded-full p-2 bg-white cursor-pointer'
                  // onClick={() => {
                  //   if (PageAction.running) {
                  //     stop();
                  //   } else {
                  //     start();
                  //   }
                  // }}
                >
                  {PageAction.running ? (
                    <SquareIcon className='size-6' />
                  ) : (
                    <PlayIcon className='size-6' />
                  )}
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
            <div className='w-1 rounded h-6 bg-black/20'></div>
            <div className='text-sm'>
              <div className='text-black/50 text-xs/4 px-2'>模拟轮次</div>
              <input
                className='block focus:outline-none w-16 px-2 h-6 rounded hover:bg-black/20'
                defaultValue={round}
              />
            </div>
            <div className='text-sm'>
              <div className='text-black/50 text-xs/4 px-2'>活跃人数</div>
              <input
                className='block focus:outline-none w-16 px-2 h-6 rounded hover:bg-black/20'
                defaultValue={roundAgentNum}
              />
            </div>
            <div className='w-1 rounded h-6 bg-black/20'></div>
            <div className='divide-x divide-black/5 text-black/75'>
              {USER_ACTION.map((action, actionIdx) => (
                <button
                  disabled
                  key={`user_action_${actionIdx}`}
                  className='p-4 py-3 bg-white cursor-pointer transition-colors hover:bg-[#0055c8] hover:text-white first:rounded-l-full first:ps-5 last:rounded-r-full last:pr-5'
                  onClick={action.onClick}
                  title={action.title}
                >
                  {action.icon}
                </button>
              ))}
            </div>
          </GlowBorderUnit>
        </div>

        <SimuGraph />

        <SimuConsole
          active={showConsole}
          onToggle={() => setShowConsole(!showConsole)}
        />
      </div>
    </>
  );
});

export default SimuMonitor;
