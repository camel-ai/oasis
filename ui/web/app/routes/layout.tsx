import clsx from 'clsx';
import { memo } from 'react';
import { FormattedMessage as F } from 'react-intl';
import { Outlet, NavLink } from 'react-router';

export default function Layout() {
  return (
    <main className='relative min-h-screen bg-gradient-to-br from-indigo-200 via-purple-100 to-emerald-200 flex flex-col'>
      <Header />
      <div className='flex-1 overflow-hidden relative'>
        <Outlet />
      </div>
    </main>
  );
}

const Header = memo(function Header() {
  const menus = [
    {
      text: 'Simulation',
      url: '/simulation',
    },
    // {
    //   text: 'History',
    //   url: '/history',
    // },
  ];

  return (
    <>
      <div className='w-full p-2 flex-none'>
        <div className='bg-white/50 backdrop-blur flex items-center rounded'>
          <div className='px-12'>
            LOGO <F id='HEADER.BRAND' />
          </div>
          {menus.map((menu, menuIdx) => (
            <NavLink
              key={`menu_${menuIdx}`}
              to={menu.url}
              className={({ isActive, isPending }) =>
                clsx(
                  'py-4 px-6 border-b-2 border-transparent hover:border-b-[#0055c8] transition-colors',
                  {
                    pending: isPending,
                    'border-b-[#0055c8]': isActive,
                  }
                )
              }
            >
              {menu.text}
            </NavLink>
          ))}
        </div>
      </div>
    </>
  );
});
