import {
  ENV_AGENT_INTERESTS,
  ENV_AGENT_PARAMS,
  ENV_AGENT_PROMPTS,
  type typeEnvAgentConfig,
  type typeEnvSettings,
} from 'app/assets/data/env_configs';
import { createContext, useContext, useRef, useState } from 'react';

const SimulationContext = createContext<any>(null);
export function useSimulationContext() {
  return useContext(SimulationContext);
}
export const SimulationConsumer = SimulationContext.Consumer;

const SimulationProvider = ({ PageEvents, children }: any) => {
  const [running, setRunning] = useState(false);
  const [graphData, setGraphData] = useState<any>({
    nodes: [],
    links: [],
    categories: [],
  });
  // TODO: will be removed
  const [consoleData, setConsoleData] = useState<any[]>([]);

  // config for agents
  const [envConfig, setEnvConfig] =
    useState<typeEnvAgentConfig[]>(ENV_AGENT_PARAMS);
  const [envInterests, setEnvInterests] = useState<any[]>(ENV_AGENT_INTERESTS);
  const [envPrompts, setEnvPrompts] = useState<string>(ENV_AGENT_PROMPTS);

  // settings for env
  const [envSettings, setEnvSettings] = useState<typeEnvSettings>({
    agent_count: 10,
    social_media_style: 'reddit',
    actions_scope: [],
  });

  const consoleRef = useRef<string>('');
  const appendConsole = (value: string) => (consoleRef.current += value);
  const getConsole = () => consoleRef.current;
  const clearConsole = () => (consoleRef.current = '');

  const value = {
    running,
    setRunning,
    graphData,
    setGraphData,
    consoleData,
    setConsoleData,
    envConfig,
    setEnvConfig,
    envInterests,
    setEnvInterests,
    envPrompts,
    appendConsole,
    getConsole,
    clearConsole,
    setEnvPrompts,
    envSettings,
    setEnvSettings,
    ...PageEvents,
  };

  return (
    <SimulationContext.Provider value={value}>
      {children}
    </SimulationContext.Provider>
  );
};

export default SimulationProvider;
