import { memo, useEffect, useMemo, useRef, useState } from 'react';

import { FormattedMessage as F } from 'react-intl';
import {
  ArrowLeftIcon,
  ChevronRight,
  LockIcon,
  LockOpenIcon,
  Trash2Icon,
  X,
} from 'lucide-react';
import pkg from 'lodash';
const { debounce } = pkg;

import { nanoid } from 'nanoid';
import clsx from 'clsx';
import {
  TooltipProvider,
  Tooltip,
  TooltipTrigger,
  TooltipContent,
  TooltipPortal,
} from '@radix-ui/react-tooltip';
import { useSimulationContext } from '../simulate-context';
import type { typeEnvAgentConfig } from 'app/assets/data/env_configs';
import { Masonry } from 'masonic';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from 'app/components/ui/alert-dialog';

const SimuAgentConfig = ({
  className,
  next,
  back,
}: {
  className?: string[] | string;
  next: () => void;
  back: () => void;
}) => {
  const SimulationContext = useSimulationContext();
  const SimulationAction = useMemo(() => {
    return {
      envConfig: SimulationContext.envConfig,
      setEnvConfig: SimulationContext.setEnvConfig,
    };
  }, [SimulationContext]);

  const [panels, setPanels] = useState([...SimulationAction.envConfig]);
  const addpanel = () => {
    setPanels([
      ...panels,
      {
        title: 'Add a new config',
        controls: [],
        defaults: [],
      },
    ]);
  };
  const removepanel = (index: number) => {
    setPanels([...panels.filter((_, i) => i !== index)]);
  };

  const panelUpdate = (value: any, index: number) => {
    panels[index] = value;
    // setPanels([...panels]);
  };

  const [isClient, setIsClient] = useState(false);
  useEffect(() => {
    setIsClient(true);
  }, []);

  const update_configs = () => {
    SimulationAction.setEnvConfig(panels);
    next();
  };

  return (
    <>
      <div className={clsx('flex flex-col w-full h-full', className)}>
        <div className='flex items-center gap-x-4 p-4 flex-none'>
          <button
            className='bg-white p-2 rounded cursor-pointer'
            onClick={back}
          >
            <ArrowLeftIcon className='size-4' />
          </button>
          <F id='SETTINGS.AGENTS.ADV_SETTINGS' />
          <button
            className='p-2 ms-auto rounded px-4 bg-[#0055c8] text-white text-sm cursor-pointer flex gap-x-2 items-center'
            onClick={update_configs}
          >
            <F id='SETTINGS.AGENTS.NEXT' />
            <ChevronRight className='inline-block size-4' />
          </button>
        </div>
        <div className='flex-1 overflow-y-auto relative'>
          <div className={clsx('px-4', className)}>
            {isClient && (
              <Masonry
                key={nanoid()}
                columnCount={3}
                rowGutter={12}
                columnGutter={12}
                items={[...panels, { role: 'add' }]}
                render={({ data: panel, index: panelIdx, width }) => (
                  <>
                    {panel.hasOwnProperty('role') && panel.role === 'add' ? (
                      <button
                        className='py-1.5 bg-black/10 cursor-pointer rounded w-full'
                        onClick={addpanel}
                      >
                        New Config
                      </button>
                    ) : (
                      <ConfigPanel
                        config={panel}
                        onUpdate={(e) => {
                          panelUpdate(e, panelIdx);
                        }}
                        onDelete={() => {
                          removepanel(panelIdx);
                        }}
                      />
                    )}
                  </>
                )}
              />
            )}
          </div>
        </div>
      </div>
    </>
  );
};

const ConfigPanel = memo(function ConfigPanel({
  config,
  onUpdate,
  onDelete,
}: {
  config: typeEnvAgentConfig;
  onUpdate: (e: any) => void;
  onDelete: (e?: any) => void;
}) {
  const [panel, setPanel] = useState<typeEnvAgentConfig>({
    ...config,
  });
  const [editable, setEditable] = useState(!config.lock && !config.saved);
  useEffect(() => {
    const len = config.controls.length;
    let keys = [];
    for (let i = 0; i < len - 1; i++) {
      keys[i] = nanoid();
    }
    setPanel({ ...config, keys });
    // if (len == 0) addcontrol();
  }, []);

  const [values, setValues] = useState<(number | undefined)[]>(config.defaults);
  const total = useMemo(() => {
    const result = values.reduce((p, c) => (p ?? 0) + (c ?? 0), 0);
    // console.log(values, result);
    return result;
  }, [values]);
  const updateValue = (
    type: 'control' | 'default' | 'title',
    value: any,
    index?: number | undefined
  ) => {
    if (type === 'title') {
      panel.title = value;
    }
    if (type === 'control' && index !== undefined) {
      panel.controls[index] = value;
    }
    if (type === 'default' && index !== undefined) {
      panel.defaults[index] = value;
      values[index] = value;
      setValues([...values]);
    }
    const _panel = { ...panel };
    setPanel(_panel);
    onUpdate(_panel);
  };

  const addcontrol = () => {
    const _panel = {
      ...panel,
      controls: [...panel.controls, undefined],
      defaults: [...panel.defaults, undefined],
      keys: [...(panel.keys ?? []), nanoid()],
    };
    setPanel(_panel);
    onUpdate(_panel);
  };

  const removecontrol = (index: number) => {
    panel.controls.splice(index, 1);
    panel.defaults.splice(index, 1);
    (panel.keys ?? []).splice(index, 1);
    const _panel = { ...panel };
    setPanel(_panel);
    onUpdate(_panel);
  };

  return (
    <>
      <div
        className={clsx('rounded bg-white/50 border text-sm', {
          'border-white': !editable,
          'border-[#0055c8]': editable,
        })}
      >
        <div className='p-2 font-bold border-b border-white bg-white rounded-t flex gap-x-2 overflow-hidden'>
          <label
            className='flex-none px-2 rounded hover:bg-black/10 flex items-center cursor-pointer'
            role='button'
          >
            <input
              type='checkbox'
              disabled={panel.lock || editable}
              defaultChecked={panel.lock || panel.checked}
              onChange={(e) => {
                onUpdate({ ...panel, checked: e.target.checked });
              }}
            />
          </label>
          {panel.lock ? (
            <div className='flex-1 h-8 flex items-center'>{panel.title}</div>
          ) : (
            <>
              <input
                className='flex-1 focus:outline-none rounded py-1.5 px-3 bg-black/5 hover:bg-black/10 w-full'
                placeholder='title'
                defaultValue={panel.title}
                disabled={!editable}
                onChange={debounce((e) => {
                  updateValue('title', e.target.value);
                }, 300)}
              />
            </>
          )}
          {!panel.lock && editable && (
            <AlertDialog>
              <AlertDialogTrigger asChild>
                <button className='flex-none rounded hover:bg-black/10 text-red-500 px-2 cursor-pointer'>
                  <Trash2Icon className='size-4' />
                </button>
              </AlertDialogTrigger>
              <AlertDialogContent className='w-96'>
                <AlertDialogHeader>
                  <AlertDialogTitle>Are you absolutely sure?</AlertDialogTitle>
                  <AlertDialogDescription>
                    Delete this Configuration Panel Data?
                    <br />
                    &nbsp;
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel>Cancel</AlertDialogCancel>
                  <AlertDialogAction onClick={onDelete}>
                    Confirm
                  </AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          )}
          <button
            className='flex-none rounded hover:bg-black/10 px-2 cursor-pointer'
            onClick={() => {
              onUpdate({ ...panel, saved: editable });
              setEditable(!editable);
            }}
          >
            {editable ? (
              <LockOpenIcon className='size-4' />
            ) : (
              <LockIcon className='size-4' />
            )}
          </button>
        </div>
        <div className='p-2 flex flex-col gap-y-2'>
          {panel.controls.map((control, controlIdx) => (
            <PanelFormInstance
              key={`control_${panel.keys?.[controlIdx] ?? controlIdx}`}
              control={control}
              defaultVal={
                panel.defaults.length > controlIdx
                  ? panel.defaults[controlIdx]
                  : undefined
              }
              onChange={(type, val) => {
                updateValue(type, val, controlIdx);
              }}
              editable={editable}
              lock={panel.lock}
              onRemove={() => removecontrol(controlIdx)}
            />
          ))}
          {!panel.lock && editable && (
            <button
              className='py-1.5 bg-black/10 cursor-pointer rounded w-full'
              onClick={addcontrol}
            >
              Add
            </button>
          )}
        </div>
        <div className='p-2 bg-white rounded-b'>
          Total
          <div className='float-right'>{(total ?? 0).toFixed(3)}</div>
        </div>
      </div>
    </>
  );
});

const PanelFormInstance = memo(function PanelFormInstance({
  onChange,
  onRemove,
  control,
  defaultVal,
  editable = false,
  lock = false,
}: {
  onChange: (t: 'control' | 'default', e?: string | number) => void;
  onRemove: () => void;
  control?: string;
  defaultVal?: number;
  editable?: boolean;
  lock?: boolean;
}) {
  return (
    <>
      <div className='flex items-center gap-x-2'>
        <div className='basis-1/3 overflow-hidden'>
          {!lock && editable ? (
            <input
              className='w-full focus:outline-none rounded py-1.5 px-3 bg-black/5 hover:bg-black/10'
              placeholder='name'
              defaultValue={control}
              onChange={debounce((e) => {
                onChange('control', e.target.value);
              }, 300)}
            />
          ) : (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <p className='line-clamp-1'>{control}</p>
                </TooltipTrigger>
                <TooltipPortal>
                  <TooltipContent className='bg-black/75 rounded text-white z-[1000]'>
                    <p className='p-2'>{control}</p>
                  </TooltipContent>
                </TooltipPortal>
              </Tooltip>
            </TooltipProvider>
          )}
        </div>
        <div className='flex-1'>
          <input
            type='number'
            className='w-full focus:outline-none rounded py-1.5 px-3 bg-black/5 hover:bg-black/10'
            placeholder='ratio'
            disabled={!editable}
            defaultValue={defaultVal}
            onChange={debounce((e) => {
              onChange('default', +e.target.value);
            }, 300)}
          />
        </div>
        {!lock && editable && (
          <div className='flex-none self-stretch'>
            <button
              className='h-full px-2 cursor-pointer hover:bg-black/10 rounded'
              onClick={onRemove}
            >
              <X className='size-4' />
            </button>
          </div>
        )}
      </div>
    </>
  );
});

export default SimuAgentConfig;
