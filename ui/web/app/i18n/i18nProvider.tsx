import { useState, useEffect, useContext, createContext } from 'react';
import { IntlProvider } from 'react-intl';
import en from './locales/en.json';
import zh from './locales/zh.json';
import zhTW from './locales/zh-TW.json';

const messages = {
  en,
  zh,
  'zh-TW': zhTW,
};

// const I18nProvider = ({ children }) => {
//   const [locale, setLocale] = useState('en'); // Default locale
//   const [currentMessages, setCurrentMessages] = useState(messages[locale]);

//   useEffect(() => {
//     setCurrentMessages(messages[locale]);
//   }, [locale]);

//   const switchLocale = (newLocale) => {
//       setLocale(newLocale);
//   };

//   return (
//     <IntlProvider locale={locale} messages={currentMessages}>
//       {children}
//     </IntlProvider>
//   );
// };

// export default I18nProvider;

const I18nContext = createContext<any>(null);
export function usePageContext() {
  return useContext(I18nContext);
}
export const I18nConsumer = I18nContext.Consumer;

const I18nProvider = ({ children }: any) => {
  const [locale, setLocale] = useState<'en' | 'zh' | 'zh-TW'>('en'); // Default locale
  const [currentMessages, setCurrentMessages] = useState<any>(messages[locale]);

  useEffect(() => {
    setCurrentMessages(messages[locale]);
  }, [locale]);

  const switchLocale = (newLocale: any) => {
    setLocale(newLocale);
  };

  const value = {
    switchLocale,
  };

  return (
    <I18nContext.Provider value={value}>
      <IntlProvider locale={locale} messages={currentMessages}>
        {children}
      </IntlProvider>
    </I18nContext.Provider>
  );
};

export default I18nProvider;
