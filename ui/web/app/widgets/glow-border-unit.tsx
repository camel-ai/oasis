import { memo } from 'react';
import clsx from 'clsx';
import './glow-border-unit.css';

const GlowBorderUnit = memo(function GlowBorderUnit({
  containerClass,
  bodyClass,
  children,
}: {
  containerClass?: string | string[];
  bodyClass?: string | string[];
  children: any;
}) {
  const onGlowDivMove = (e: any) => {
    e.stopPropagation();
    e.preventDefault();
    const rect = e.currentTarget.getBoundingClientRect();

    e.currentTarget.style.setProperty(
      '--glow-left',
      e.clientX - rect.left + 'px'
    );
    e.currentTarget.style.setProperty(
      '--glow-top',
      e.clientY - rect.top + 'px'
    );
  };
  return (
    <div
      className={clsx(
        'glow-border-unit group/glow relative overflow-hidden',
        containerClass
      )}
      onMouseMove={onGlowDivMove}
    >
      <div
        className={clsx(
          `glow-border absolute inset-0 box-border p-px opacity-0 transition-opacity duration-500 
           before:content[''] before:absolute before:h-full before:max-h-[1/3] before:w-full before:max-w-[1/3] before:aspect-square
           before:translate-x-[-50%] before:translate-y-[-50%] 
           before:bg-radial before:from-white before:to-transparent 
           before:to-50% group-hover/glow:opacity-100`
        )}
      ></div>
      <div className={clsx('glow-content relative z-20 p-px', bodyClass)}>
        {children}
      </div>
    </div>
  );
});

export default GlowBorderUnit;
