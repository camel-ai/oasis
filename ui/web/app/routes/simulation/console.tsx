import clsx from 'clsx';
import { memo, useMemo, useRef, useEffect, useState } from 'react';
import { useSimulationContext } from './simulate-context';

const SimuConsole = memo(function SimuConsole({
  active,
  onToggle,
}: {
  active: boolean;
  onToggle: () => void;
}) {
  const SimulationContext = useSimulationContext();
  const SimulationAction = useMemo(() => {
    return {
      consoleData: SimulationContext.consoleData,
      getConsole: SimulationContext.getConsole,
    };
  }, [SimulationContext]);

  const [message, setMessage] = useState('');
  const containerRef = useRef<HTMLDivElement>(null);

  const animationRef = useRef<any>(null);
  useEffect(() => {
    if (!active) return;
    let startTime = 0;
    const animate = (timestamp: any) => {
      const elapsed = timestamp - startTime;

      if (elapsed > 1000) {
        setMessage(SimulationAction.getConsole());
        startTime = timestamp;
      }
      animationRef.current = requestAnimationFrame(animate);
    };
    animationRef.current = requestAnimationFrame(animate);
    return () => {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
      containerRef.current = null;
    };
  }, [active]);

  useEffect(() => {
    const container = containerRef.current;
    if (container) {
      requestAnimationFrame(() => {
        container.scrollTop = container.scrollHeight;
      });
    }
  }, [message]);

  return (
    <>
      <div
        className={clsx(
          'absolute bottom-0 left-0 z-10 px-4 w-full transition-transform',
          {
            'translate-y-0': active,
            'translate-y-[100%]': !active,
          }
        )}
      >
        <div className='absolute top-0 translate-y-[-100%]'>
          <button
            onClick={onToggle}
            className='rounded-t bg-white cursor-pointer py-1.5 px-4'
          >
            Console
          </button>
        </div>
        <div className='bg-white p-2 rounded-tr'>
          <div
            ref={containerRef}
            className='rounded bg-black/5 text-sm/6 p-2 h-56 overflow-y-auto text-black/75'
            dangerouslySetInnerHTML={{ __html: message }}
          />
        </div>
      </div>
    </>
  );
});

export default SimuConsole;
