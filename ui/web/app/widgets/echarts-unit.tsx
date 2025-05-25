import * as echarts from 'echarts';
import { useRef, useEffect, memo } from 'react';

// echarts.use([CanvasRenderer]);
// type EChartOption = echarts.EChartOption;
type EChartOption = any;
// type EChartsResponsiveOption = echarts.EChartsResponsiveOption;
type EChartsResponsiveOption = any;

// const defaultOptions = {
//   grid: {
//     top: 20,
//     bottom: 20,
//     left: 30,
//     right: 10,
//   },
// };

const EchartUnit = ({
  defaultOptions,
  options,
  onXAxisClicked,
}: {
  defaultOptions: EChartOption;
  options?: EChartOption | EChartsResponsiveOption;
  onXAxisClicked?: (e: any) => void;
}) => {
  const canvasRef = useRef<any>(null);
  const canvasContainerRef = useRef<any>(null);
  const chartInstRef = useRef<any>(null);

  useEffect(() => {
    const containerElem = canvasContainerRef.current;
    const chartElem = canvasRef.current!;

    const chartInst = echarts.init(chartElem, null, {
      width: containerElem.offsetWidth,
      height: containerElem.offsetHeight,
      renderer: 'canvas',
      useDirtyRect: false,
    });
    chartInst.setOption(defaultOptions);
    chartInstRef.current = chartInst;

    chartInstRef.current.on('click', (params: any) => {
      console.log(params);
      onXAxisClicked && onXAxisClicked(params);
      // console.log(dataAxis[Math.max(params.dataIndex - zoomSize / 2, 0)]);
    });

    const handleResize = () => {
      chartElem.style.width = containerElem.offsetWidth + 'px';
      chartElem.style.height = containerElem.offsetHeight + 'px';
      chartInst.resize();
    };
    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, []);

  useEffect(() => {
    // chartInstRef.current.setOption({ ...defaultOptions, ...options });
    console.log(options);
    chartInstRef.current.setOption({ ...options });
    // chartInstRef.current.
  }, [options]);

  return (
    <div className='absolute inset-0' ref={canvasContainerRef}>
      <canvas ref={canvasRef} />
    </div>
  );
};

export default EchartUnit;
