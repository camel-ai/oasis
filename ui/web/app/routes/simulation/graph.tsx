import EchartUnit from 'app/widgets/echarts-unit';
import { memo, useEffect, useMemo, useRef, useState } from 'react';

import { generateGraph } from 'app/utils';

import { useSimulationContext } from './simulate-context';

const SimuGraph = memo(function SimuGraph() {
  const SimulationContext = useSimulationContext();
  const PageAction = useMemo(() => {
    return {
      graphData: SimulationContext.graphData,
    };
  }, [SimulationContext]);

  const options = useMemo(() => {
    const result = PageAction.graphData;
    return {
      legend: [
        {
          data: (result.categories ?? []).map((a: any) => {
            return a.name;
          }),
          bottom: 5,
        },
      ],
      series: [
        {
          data: result.nodes ?? [],
          links: result.links ?? [],
          categories: result.categories ?? [],
        },
      ],
    };
  }, [PageAction.graphData]);

  return (
    <EchartUnit
      options={options}
      defaultOptions={{
        series: [
          {
            type: 'graph',
            layout: 'force',
            // animation: false,
            data: [
              {
                fixed: true,
                x: 20,
                y: 20,
                symbolSize: 20,
                id: '-1',
              },
            ],
            links: [],
            force: {
              // initLayout: 'circular'
              // gravity: 0
              repulsion: 50,
              edgeLength: 5,
            },
            roam: true,
            draggable: true,
            label: {
              show: true,
              position: 'right',
              formatter: '{b}',
            },
            labelLayout: {
              hideOverlap: true,
            },
            // scaleLimit: {
            //   min: 0.4,
            //   max: 2,
            // },
            lineStyle: {
              color: 'source',
              curveness: 0.3,
            },
            emphasis: {
              focus: 'adjacency',
              lineStyle: {
                width: 10,
              },
            },
          },
        ],
        // title: {},
        // tooltip: {},
        // legend: { show: false },
        // animationDuration: 1500,
        // animationEasingUpdate: 'quinticInOut',
        // series: [
        //   {
        //     name: 'Les Miserables',
        //     type: 'graph',
        //     legendHoverLink: false,
        //     layout: 'none',
        //     data: graph_data.nodes ?? [],
        //     links: graph_data.links ?? [],
        //     categories: graph_data.categories ?? [],
        //     roam: true,
        //     label: {
        //       position: 'right',
        //       formatter: '{b}',
        //       show: true,
        //     },
        //     // force: {
        //     //   repulsion: 200,
        //     // },
        //     lineStyle: {
        //       color: 'source',
        //       curveness: 0.3,
        //     },
        //     emphasis: {
        //       focus: 'adjacency',
        //       lineStyle: {
        //         width: 10,
        //       },
        //     },
        //   },
        // ],
      }}
    />
  );
});

export default SimuGraph;
