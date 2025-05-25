import {
  NavLink,
  Outlet,
  useLocation,
  type MetaDescriptor,
} from 'react-router';
import { StarIcon } from 'lucide-react';

import { useEffect, useState } from 'react';
import { SSE } from 'sse.js';
import type { AppService } from 'src/app.service';

import type { Route } from './+types/home';
import clsx from 'clsx';

export function meta() {
  return [
    { title: 'React Router 7 + NestJS Custom Server • cbnsndwch OSS' },
  ] satisfies MetaDescriptor[];
}

export async function loader({ context }: LoaderFunctionArgs) {
  // const appService = context.app.get<AppService>('AppService');

  return {
    // hello: appService.getHello(),
    hello: 'hello world',
  };
}

export default function Home({ loaderData }: Route.ComponentProps) {
  const { hello } = loaderData;
  // const user = useOptionalUser();

  const [message, setMessage] = useState('');
  // useEffect(() => {
  //   const source = new SSE('/api/calc');
  //   source.onmessage = ({ data }) => {
  //     // Assuming we receive JSON-encoded data payloads:
  //     const result = JSON.parse(data);
  //     setMessage((message) => {
  //       return message + (result.output ?? '') + '<br />';
  //     });
  //   };

  //   // eventSource.onmessage = ({ data }) => {
  //   //   const eventData = JSON.parse(data);
  //   //   console.log('New message', eventData);
  //   //   setPlcData(eventData);

  //   //   if (!currentProduct && eventData.product) {
  //   //     setCurrentProduct({
  //   //       value: eventData.product.id,
  //   //       label: eventData.product.name,
  //   //     });
  //   //   }
  //   // };
  //   // eventSource.onerror = (err) => {
  //   //   console.log('err', err);
  //   // };
  //   // eventSource.onopen = (open) => {
  //   //   console.log('open', open);
  //   // };

  //   return () => source.close();
  // }, []);

  return (
    <main className=''>
      hi five home
      {/* <div
        className='h-48 overflow-y-auto'
        dangerouslySetInnerHTML={{ __html: message }}
      /> */}
    </main>
  );
}
