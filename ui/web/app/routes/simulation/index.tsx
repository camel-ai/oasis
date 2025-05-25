import clsx from 'clsx';
import {
  ArrowLeftIcon,
  ChevronRight,
  InfoIcon,
  Settings,
  Settings2Icon,
  X,
} from 'lucide-react';
import { memo, useMemo, useState } from 'react';

import SimuMonitor from './monitor';
import SimulationProvider, { useSimulationContext } from './simulate-context';
import GlowBorderUnit from 'app/widgets/glow-border-unit';
import { FormattedMessage as F } from 'react-intl';
import SimuAgentIndex from './agents';
import SimuAgentList from './agents/list';
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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from 'app/components/ui/select';
import { Textarea } from 'app/components/ui/textarea';
import { Input } from 'app/components/ui/input';
import pkg from 'lodash';
import type { typeEnvSettings } from 'app/assets/data/env_configs';
const { debounce } = pkg;

const AVALIABLE_ACTIONS = [
  'SETTINGS.ENV.ACTIONS.LIKE',
  'SETTINGS.ENV.ACTIONS.UNLIKE',
  'SETTINGS.ENV.ACTIONS.POST',
  'SETTINGS.ENV.ACTIONS.COMMENT',
  'SETTINGS.ENV.ACTIONS.LIKE_COMMENT',
  'SETTINGS.ENV.ACTIONS.DISLIKE_COMMENT',
  'SETTINGS.ENV.ACTIONS.SEARCH_POSTS',
  'SETTINGS.ENV.ACTIONS.SEARCH_USER',
  'SETTINGS.ENV.ACTIONS.TREND',
  'SETTINGS.ENV.ACTIONS.REFRESH',
  'SETTINGS.ENV.ACTIONS.DO_NOTHING',
  'SETTINGS.ENV.ACTIONS.FOLLOW',
  'SETTINGS.ENV.ACTIONS.MUTE',
];

export default function Simulation() {
  const [current, setCurrentStep] = useState('setting');

  const PageEvents = {
    // startRunning: () => {
    //   console.log('add');
    // },
  };

  return (
    <>
      <SimulationProvider PageEvents={PageEvents}>
        <div className='absolute inset-0 p-2'>
          <SimuSettings
            active={current == 'setting'}
            onNext={(reset) => {
              setCurrentStep('simulate');
            }}
            onBack={() => setCurrentStep('setting')}
          />
          <SimuMonitor active={current == 'simulate'} />
        </div>
      </SimulationProvider>
    </>
  );
}

const SimuSettings = memo(function SimuSettings({
  active,
  onNext,
  onBack,
}: {
  active: boolean;
  onNext: (reset?: boolean) => void;
  onBack: () => void;
}) {
  const SimulationContext = useSimulationContext();
  const SimulationAction = useMemo(() => {
    return {
      envSettings: SimulationContext.envSettings,
      setEnvSettings: SimulationContext.setEnvSettings,
    };
  }, [SimulationContext]);

  const [showUser, setShowUser] = useState(false);
  const [advancedUser, setAdvancedUser] = useState(0);

  const [settings, setSettings] = useState(SimulationAction.envSettings);

  const updateSettings = (name: string, value: any) => {
    SimulationAction.setEnvSettings((settingsValue: typeEnvSettings) => ({
      ...settingsValue,
      [name]: value,
    }));
  };

  return (
    <>
      {active && (
        <>
          <div
            className={clsx(
              'absolute left-1/4 top-0 p-2 px-4 h-full transition-transform transition-opacity w-2/3',
              {
                'translate-x-0 opacity-100 z-10': showUser,
                'translate-x-[50%] opacity-0 -z-10': !showUser,
              }
            )}
          >
            <SimuAgentList back={() => setShowUser(false)} />
          </div>
          <div
            className={clsx(
              'absolute left-1/4 top-0 p-2 px-4 h-full transition-transform transition-opacity w-2/3',
              {
                'translate-x-0 opacity-100 z-10': advancedUser > 0,
                'translate-x-[50%] opacity-0 -z-10': advancedUser == 0,
              }
            )}
          >
            <SimuAgentIndex
              current={advancedUser}
              changeIndex={setAdvancedUser}
            />
          </div>
        </>
      )}
      <div
        className={clsx(
          'rounded bg-white/50 border border-white w-1/4 relative h-full flex flex-col transition-transform z-20',
          {
            'translate-x-0': active,
            'translate-x-[-100%]': !active,
          }
        )}
      >
        {!active && (
          <div className='absolute right-0 top-0 translate-x-[100%] px-2'>
            <button
              className='bg-white/50 p-2 rounded border border-white cursor-pointer flex gap-x-2 items-center text-sm'
              onClick={onBack}
            >
              Config
              <Settings2Icon className='size-4' />
            </button>
          </div>
        )}
        <div className='flex items-center gap-x-4 p-4 flex-none'>
          <button
            className='bg-white p-2 rounded cursor-pointer'
            onClick={() => onNext()}
          >
            <ArrowLeftIcon className='size-4' />
          </button>
          Simulation Env Setting
          <AlertDialog>
            <AlertDialogTrigger asChild>
              <button
                className='p-2 ms-auto rounded px-4 bg-[#0055c8] text-white text-sm cursor-pointer'
                onClick={() => {
                  // onNext(true);
                }}
              >
                Play Sim
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
        <div className='flex-1 px-4 overflow-y-auto'>
          <div className='mt-4 text-black/75'>
            <div className='font-bold pb-2 flex items-center gap-x-2'>
              <F id='SETTINGS.ENV.AGENTS.TITLE' />
              <div className='ms-auto flex gap-x-2'>
                <button
                  className='text-[#0055c8] cursor-pointer'
                  onClick={() => {
                    setShowUser(true);
                    setAdvancedUser(0);
                  }}
                >
                  <F id='SETTINGS.ENV.AGENTS.ACTIONS.PREVIEW' />
                </button>
                <button
                  className='text-[#0055c8] cursor-pointer'
                  onClick={() => {
                    setAdvancedUser(1);
                    setShowUser(false);
                  }}
                >
                  <F id='SETTINGS.ENV.AGENTS.ACTIONS.ADVANCED' />
                </button>
              </div>
            </div>
            <div className='pb-8'>
              <Input
                type='number'
                placeholder='input a number'
                className='w-full rounded bg-black/5 px-3 py-2'
                defaultValue={settings.agent_count}
                onChange={debounce((e) => {
                  updateSettings('agent_count', +e.target.value);
                }, 300)}
              />
            </div>
            <div className='font-bold inline-flex gap-x-1 items-center pb-2'>
              <F id='SETTINGS.ENV.SOCIALMEDIA.TITLE' />
              <InfoIcon className='size-4' />
            </div>
            <div className='pb-8'>
              <Select defaultValue={settings.social_media_style}>
                <SelectTrigger className='w-full bg-black/5 rounded'>
                  <SelectValue placeholder='Social Media' />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value='reddit'>Reddit</SelectItem>
                  <SelectItem value='twitter'>Twitter</SelectItem>
                </SelectContent>
              </Select>
              <div className='pt-2 text-black/50 text-sm/4'>
                <F id='SETTINGS.ENV.SOCIALMEDIA.INFO' />
              </div>
            </div>
            <div className='font-bold inline-flex gap-x-1 items-center pb-2'>
              <F id='SETTINGS.ENV.AVAILABLEACTIONS.TITLE' />
              <InfoIcon className='size-4' />
            </div>
            <div className='pb-8'>
              <div className='flex gap-2 flex-wrap'>
                {AVALIABLE_ACTIONS.map((action, actionIdx) => (
                  <button
                    key={`action_${actionIdx}`}
                    className='text-sm rounded p-1 px-1.5 rounded bg-black/75 text-white flex gap-x-1.5 items-center cursor-pointer'
                  >
                    <F id={action} />
                    <X className='size-4' />
                  </button>
                ))}
                <button className='text-sm rounded p-1 px-3 rounded bg-white text-black/75 cursor-pointer'>
                  +
                </button>
              </div>
              <div className='pt-2 text-black/50 text-sm/4'>
                <F id='SETTINGS.ENV.AVAILABLEACTIONS.INFO' />
              </div>
            </div>
            <div className='font-bold inline-flex gap-x-1 items-center pb-2'>
              <F id='SETTINGS.ENV.INITIALACTIONS.TITLE' />
              <InfoIcon className='size-4' />
            </div>
            <div className='pb-8'>
              <Textarea
                className='w-full h-40 py-2 px-3 bg-black/5 hover:bg-black/10 resize-none'
                placeholder='Start from a post'
                value='Hello, world!'
                readOnly
              />
              {/* <div className='pt-2 text-black/50 text-sm/4'>
              </div> */}
            </div>
          </div>
        </div>
      </div>
    </>
  );
});
