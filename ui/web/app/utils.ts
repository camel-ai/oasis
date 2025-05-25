import { useMemo } from 'react';
import { useMatches } from 'react-router';

type User = Record<string, any>;

const DEFAULT_REDIRECT = '/';

/**
 * This should be used any time the redirect path is user-provided
 * (Like the query string on our login/signup pages). This avoids
 * open-redirect vulnerabilities.
 * @param {string} to The redirect destination
 * @param {string} defaultRedirect The redirect to use if the to is unsafe.
 */
export function safeRedirect(
  to: FormDataEntryValue | string | null | undefined,
  defaultRedirect: string = DEFAULT_REDIRECT
) {
  if (!to || typeof to !== 'string') {
    return defaultRedirect;
  }

  if (!to.startsWith('/') || to.startsWith('//')) {
    return defaultRedirect;
  }

  return to;
}

/**
 * This base hook is used in other hooks to quickly search for specific data
 * across all loader data using useMatches.
 * @param {string} id The route id
 * @returns {JSON|undefined} The router data or undefined if not found
 */
export function useMatchesData(
  id: string
): Record<string, unknown> | undefined {
  const matchingRoutes = useMatches();
  const route = useMemo(
    () => matchingRoutes.find((route) => route.id === id),
    [matchingRoutes, id]
  );
  return route?.data as Record<string, unknown>;
}

function isUser(user: unknown): user is User {
  return (
    user != null &&
    typeof user === 'object' &&
    'email' in user &&
    typeof user.email === 'string'
  );
}

export function useOptionalUser(): User | undefined {
  const data = useMatchesData('root');
  if (!data || !isUser(data.user)) {
    return undefined;
  }
  return data.user;
}

export function useUser(): User {
  const maybeUser = useOptionalUser();
  if (!maybeUser) {
    throw new Error(
      'No user found in root loader, but user is required by useUser. If user is optional, try useOptionalUser instead.'
    );
  }
  return maybeUser;
}

export function validateEmail(email: unknown): email is string {
  return typeof email === 'string' && email.length > 3 && email.includes('@');
}

export function generateGraph(
  items: {
    user_id: number;
    created_at: Date;
    action: string;
    info: any;
  }[],
  origin = {
    nodes: [],
    links: [],
    categories: [],
  }
) {
  const categories = [...new Set(items.map((i) => i.action))];

  const PROCESS_DATA: {
    nodes: any[];
    links: any[];
    categories: any[];
  } = {
    ...origin,
    categories: [...origin.categories.map((c: any) => c.name), ...categories],
  };

  for (let i = 0; i < items.length; i++) {
    const current_action = items[i].action;
    const params = JSON.parse(items[i].info);
    const categoryIdx = categories.findIndex((c) => c == current_action);

    let current_node: any = {
      value: params,
      category: categoryIdx,
    };
    switch (current_action) {
      case 'sign_up':
        current_node = {
          ...current_node,
          id: `user_${items[i].user_id}`,
          name: params['name'],
          symbolSize: 10,
        };
        PROCESS_DATA.nodes.push(current_node);
        break;
      case 'create_post':
        current_node = {
          ...current_node,
          id: `post_${params['post_id']}`,
          name: params['content'],
        };
        PROCESS_DATA.nodes.push(current_node);
        PROCESS_DATA.links.push({
          source: `user_${items[i].user_id}`,
          target: `post_${params['post_id']}`,
        });
        break;
      case 'create_comment':
        current_node = {
          ...current_node,
          id: `comment_${params['comment_id']}`,
          name: params['content'],
        };
        PROCESS_DATA.nodes.push(current_node);
        PROCESS_DATA.links.push(
          ...[
            {
              source: `user_${items[i].user_id}`,
              target: `comment_${params['comment_id']}`,
            },
            {
              source: `comment_${params['comment_id']}`,
              target: `post_${params['post_id']}`,
            },
          ]
        );
        break;
      case 'like_comment':
        current_node = {
          ...current_node,
          id: `comment_like_${params['comment_like_id']}`,
          name: 'like',
        };
        PROCESS_DATA.nodes.push(current_node);
        PROCESS_DATA.links.push(
          ...[
            {
              source: `user_${items[i].user_id}`,
              target: `comment_like_${params['comment_like_id']}`,
            },
            {
              source: `comment_like_${params['comment_like_id']}`,
              target: `comment_${params['comment_id']}`,
            },
          ]
        );
        break;
      case 'follow':
        break;
      case 'refresh':
      default:
        break;
    }

    // if (current_node.id) {
    //   const link_node = PROCESS_DATA.nodes.find(
    //     (n) => n.id === current_node.id
    //   );
    //   const link_target = PROCESS_DATA.links.filter(
    //     (n) => n.target === current_node.id
    //   );
    //   link_node.symbolSize = (link_target.length + 1) * 20;
    // }
  }

  PROCESS_DATA.categories = PROCESS_DATA.categories.map((c) => ({ name: c }));
  return PROCESS_DATA;
}
