import clsx from 'clsx';
import { useRef, useEffect } from 'react';

import type { ReactNode } from 'react';

const MasonryUnit = ({
  panels,
  panelWrapper,
  className,
  prefix,
  suffix,
}: {
  panels: any[];
  panelWrapper: (e: any, i: number) => ReactNode;
  className?: string[] | string;
  prefix?: ReactNode;
  suffix?: ReactNode;
}) => {
  const masonryRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const handleResize = () => {
      const items = masonryRef.current?.children;
      if (!items || items?.length == 0) return;

      Array.from(items).forEach((item) => {
        const rowHeight = parseInt(
          window
            .getComputedStyle(masonryRef.current!)
            .getPropertyValue('grid-auto-rows')
        );
        const rowGap = parseInt(
          window
            .getComputedStyle(masonryRef.current!)
            .getPropertyValue('grid-row-gap')
        );
        const rowSpan = Math.ceil(
          (item
            .querySelector(':scope > .masonry-content')!
            .getBoundingClientRect().height +
            rowGap) /
            (rowHeight + rowGap)
        );
        (item as HTMLElement).style.gridRowEnd = 'span ' + rowSpan;
      });
    };
    // handleResize();

    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [panels]);
  return (
    <div
      ref={masonryRef}
      className={clsx('grid gap-4', className)}
      style={{
        // gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
        // gridAutoRows: '6px',

        gridTemplateColumns: 'repeat(3, 1fr)',
        gridTemplateRows: 'masonry',
      }}
    >
      {prefix && (
        <div>
          <div className='masonry-content'>{prefix}</div>
        </div>
      )}
      {panels.map((panel, panelIdx) => (
        <div key={`mansory_panel_${panelIdx}`}>
          <div className='masonry-content'>{panelWrapper(panel, panelIdx)}</div>
        </div>
      ))}
      {suffix && (
        <div>
          <div className='masonry-content'>{suffix}</div>
        </div>
      )}
    </div>
  );
};

export default MasonryUnit;
