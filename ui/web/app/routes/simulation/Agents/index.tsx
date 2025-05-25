import GlowBorderUnit from 'app/widgets/glow-border-unit';
import { FormattedMessage as F } from 'react-intl';
import { useSimulationContext } from '../simulate-context';
import { useEffect, useMemo, useRef, useState } from 'react';
import { ArrowLeftIcon, ChevronRight } from 'lucide-react';
import clsx from 'clsx';
import SimuConsole from '../console';
import SimuAgentConfig from '../agents/config';
import SimuAgentPrompt from '../agents/prompt';
import { toast } from 'sonner';

import { SSE } from 'sse.js';

const advancedUserSteps = 2;

const SimuAgentIndex = ({
  current,
  changeIndex,
}: {
  current: number;
  changeIndex: (e: number) => void;
}) => {
  const SimulationContext = useSimulationContext();
  const SimulationAction = useMemo(() => {
    return {
      envConfig: SimulationContext.envConfig,
      envInterests: SimulationContext.envInterests,
      envPrompts: SimulationContext.envPrompts,
      envSettings: SimulationContext.envSettings,
      clearConsole: SimulationContext.clearConsole,
      appendConsole: SimulationContext.appendConsole,
    };
  }, [SimulationContext]);

  const [showConsole, setShowConsole] = useState(false);

  const sourceRef = useRef<SSE>(null);
  useEffect(() => {
    sourceRef.current = new SSE('/api/ipc-channel/start', {
      start: false,
    });
    sourceRef.current.onmessage = ({ data }: any) => {
      // Assuming we receive JSON-encoded data payloads:
      const result = JSON.parse(data);

      if (result.hasOwnProperty('output')) {
        SimulationContext.appendConsole(
          JSON.stringify(result['output']['result']) + '<br />'
        );
      }
      if (result.hasOwnProperty('log')) {
        SimulationContext.appendConsole(result['log'] + '<br />');
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

  const create_agents = async () => {
    SimulationAction.clearConsole();
    const agent_count = SimulationAction.envSettings.agent_count;
    const configs = SimulationAction.envConfig
      .filter((c: any) => c.lock || c.checked)
      .reduce(
        (pre: any, cur: any) => ({
          ...pre,
          [cur.title.toLowerCase()]: {
            groups: cur.controls,
            ratios: cur.defaults,
          },
        }),
        {}
      );
    const interests = SimulationAction.envInterests.reduce(
      (pre: any, cur: any) => ({
        names: [...pre.names, cur.name],
        descs: [...pre.descs, cur.desc],
      }),
      {
        names: [],
        descs: [],
      }
    );
    const params = {
      ...configs,
      ['interested topics']: interests,
    };
    try {
      const loadingtoastid = toast.loading(
        'Request sent to generate Agents, Please wait...'
      );
      setShowConsole(true);
      const response = await fetch(`/api/user/create`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          count: agent_count,
          params,
          prompts: SimulationAction.envPrompts,
        }),
      });
      const result = await response.json();
      console.log('action result: ', result);

      if (result.success) {
        toast.dismiss(loadingtoastid);
        toast.success(result.message);
      }
      // return;
    } catch (err) {
      console.log(err);
    }
  };

  return (
    <>
      <div className='pb-10 h-full'>
        <GlowBorderUnit
          containerClass='bg-white/50 rounded backdrop-blur w-full h-full'
          bodyClass='rounded w-full h-full'
        >
          <SimuAgentConfig
            next={() => {
              changeIndex(current + 1);

              sourceRef.current?.stream();
            }}
            back={() => changeIndex(current - 1)}
            className={clsx({
              flex: current == 1,
              hidden: current !== 1,
            })}
          />
          <SimuAgentPrompt
            next={create_agents}
            back={() => changeIndex(current - 1)}
            className={clsx({
              flex: current == 2,
              hidden: current !== 2,
            })}
          />
        </GlowBorderUnit>
      </div>
      <SimuConsole
        active={showConsole}
        onToggle={() => setShowConsole(!showConsole)}
      />
    </>
  );
};

export default SimuAgentIndex;
