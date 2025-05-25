import clsx from 'clsx';
import { ArrowLeftIcon, PlusIcon, Trash2Icon } from 'lucide-react';
import { memo, useMemo, useState } from 'react';
import { useSimulationContext } from '../simulate-context';
import { FormattedMessage as F } from 'react-intl';
import { Textarea } from 'app/components/ui/textarea';

import pkg from 'lodash';
const { debounce } = pkg;

const SimuAgentPrompt = memo(function SimuAgentPrompt({
  className,
  next,
  back,
}: {
  className?: string[] | string;
  next: () => void;
  back: () => void;
}) {
  const SimulationContext = useSimulationContext();
  const SimulationAction = useMemo(() => {
    return {
      envInterests: SimulationContext.envInterests,
      setEnvInterests: SimulationContext.setEnvInterests,
      envPrompts: SimulationContext.envPrompts,
      setEnvPrompts: SimulationContext.setEnvPrompts,
    };
  }, [SimulationContext]);

  const [interests, setInterests] = useState<any[]>(
    SimulationAction.envInterests
  );

  const addtopic = () => {
    const _topics = [
      ...interests,
      {
        name: '',
        desc: '',
      },
    ];
    setInterests(_topics);
    SimulationAction.setEnvInterests(_topics);
  };
  const removetopic = (topic: any) => {
    const _topics = interests.filter(
      (interest) =>
        !(interest.name == topic.name && interest.desc == topic.desc)
    );
    setInterests(_topics);
    SimulationAction.setEnvInterests(_topics);
  };

  const update_topic = (
    type: 'name' | 'desc',
    value: string,
    index: number
  ) => {
    const _topic = interests[index];
    _topic[type] = value;
    SimulationAction.setEnvInterests(interests);
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
          <F id='SETTINGS.AGENTS.CUSTOMIZE_PROMPTS' />
          <button
            className='p-2 ms-auto rounded px-4 bg-[#0055c8] text-white text-sm cursor-pointer flex gap-x-2 items-center'
            onClick={next}
          >
            <F id='SETTINGS.AGENTS.GENERATE' />
          </button>
        </div>
        <div className='flex-1 overflow-y-auto relative'>
          <div className='px-4 text-sm flex flex-col gap-y-8 pb-8'>
            <div>
              <p className='font-bold pb-2'>Interested Topics</p>
              <table className='border-collapse table-fixed w-full border border-white rounded-t'>
                <thead>
                  <tr className='bg-white'>
                    <th className='p-2 rounded-tl text-left w-48'>name</th>
                    <th className='p-2 text-left'>description</th>
                    <th className='p-2 w-12'></th>
                  </tr>
                </thead>
                <tbody>
                  {interests.map((topic, topicIdx) => (
                    <tr
                      key={`topics_${topicIdx}`}
                      className='hover:bg-black/10 even:bg-black/5'
                    >
                      <td className='p-2 h-24'>
                        {/* <div
                      contentEditable
                      data-placeholder='name'
                      className={`w-full h-full focus:outline-none rounded py-1.5 px-3 bg-black/5 hover:bg-black/10 overflow-y-auto
                              relative empty:before:content-[attr(data-placeholder)] empty:before:text-black/50 empty:before:block empty:before:pointer-events-none`}
                      suppressContentEditableWarning
                      onKeyUp={(e: any) => {
                        console.log('up', e.target.innerHTML);
                      }}
                      onChange={(e) => {
                        console.log(e);
                      }}
                    >
                      {topic.name}
                    </div> */}

                        <Textarea
                          className='w-full h-full py-1.5 px-3 bg-black/5 hover:bg-black/10 resize-none'
                          placeholder='name'
                          defaultValue={topic.name}
                          onChange={debounce((e) => {
                            update_topic('name', e.target.value, topicIdx);
                          }, 300)}
                        />
                      </td>
                      <td className='p-2 h-24' colSpan={topic.lock ? 2 : 1}>
                        {/* <div
                      contentEditable
                      data-placeholder='description'
                      className={`w-full h-full focus:outline-none rounded py-1.5 px-3 bg-black/5 hover:bg-black/10 overflow-y-auto
                              relative empty:before:content-[attr(data-placeholder)] empty:before:text-black/50 empty:before:block empty:before:pointer-events-none`}
                      suppressContentEditableWarning
                    >
                      {topic.desc}
                    </div> */}
                        <Textarea
                          className='w-full h-full py-1.5 px-3 bg-black/5 hover:bg-black/10 resize-none'
                          placeholder='description'
                          defaultValue={topic.desc}
                          onChange={debounce((e) => {
                            update_topic('desc', e.target.value, topicIdx);
                          }, 300)}
                        />
                      </td>
                      {!topic.lock && (
                        <td className='text-center align-middle'>
                          <button
                            className='h-8 rounded hover:bg-black/10 text-red-500 px-2 cursor-pointer'
                            onClick={() => removetopic(topic)}
                          >
                            <Trash2Icon className='size-4' />
                          </button>
                        </td>
                      )}
                    </tr>
                  ))}
                  <tr>
                    <td className='p-2' colSpan={3}>
                      <button
                        className='py-1.5 bg-black/10 cursor-pointer rounded w-full'
                        onClick={addtopic}
                      >
                        Add
                      </button>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
            <div>
              <p className='font-bold pb-2'>User Profile</p>
              {/* <div
            contentEditable
            data-placeholder='user prompt'
            className={`w-full h-36 outline outline-solid outline-white focus:outline-none rounded py-1.5 px-3 bg-black/5 hover:bg-black/10 overflow-y-auto
            relative empty:before:content-[attr(data-placeholder)] empty:before:text-black/50 empty:before:block empty:before:pointer-events-none`}
            suppressContentEditableWarning
          >
            Please generate a social media user profile based on the provided
            personal information, including a real name, username, user bio, and
            a new user persona. The focus should be on creating a fictional
            background story and detailed interests based on their hobbies and
            profession.
          </div> */}

              <Textarea
                className='w-full h-56 py-1.5 px-3 bg-black/5 hover:bg-black/10 resize-none'
                placeholder='user prompt'
                defaultValue={SimulationAction.envPrompts}
                onChange={debounce((e) => {
                  SimulationAction.setEnvPrompts(e.target.value);
                }, 300)}
              />
              <p className='text-black/50 pt-2'>
                Modify <b>"Prompt Message"</b> above to generate{' '}
                <b>"User Profile"</b> you want.
              </p>
            </div>
          </div>
        </div>
      </div>
    </>
  );
});

export default SimuAgentPrompt;
