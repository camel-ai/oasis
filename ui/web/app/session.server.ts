import { createCookieSessionStorage, redirect } from 'react-router';
import invariant from 'tiny-invariant';

type User = Record<string, any>;
const users = new Map<string, User>([
    ['1', { id: '1', username: 'john_doe' }],
    ['2', { id: '2', username: 'jane_doe' }]
]);

invariant(process.env.SESSION_SECRET, 'SESSION_SECRET must be set');

export const sessionStorage = createCookieSessionStorage({
    cookie: {
        name: '__session',
        httpOnly: true,
        path: '/',
        sameSite: 'lax',
        secrets: [process.env.SESSION_SECRET],
        secure: process.env.NODE_ENV === 'production'
    }
});

const USER_SESSION_KEY = 'userId';

export async function getSession(request: Request) {
    const cookie = request.headers.get('Cookie');
    return sessionStorage.getSession(cookie);
}

export async function getUserId(
    request: Request
): Promise<User['id'] | undefined> {
    const session = await getSession(request);
    const userId = session.get(USER_SESSION_KEY);
    return userId;
}

export async function getUser(request: Request) {
    const userId = await getUserId(request);
    if (userId === undefined) return null;

    const user = users.get(userId);
    if (user) {
        return user;
    }

    throw await logout(request);
}

export async function assertUserId(
    request: Request,
    redirectTo: string = new URL(request.url).pathname
) {
    const userId = await getUserId(request);
    if (!userId) {
        const searchParams = new URLSearchParams([['redirectTo', redirectTo]]);
        throw redirect(`/login?${searchParams}`);
    }
    return userId;
}

export async function assertUser(request: Request) {
    const userId = await assertUserId(request);

    const user = users.get(userId);
    if (user) {
        return user;
    }

    throw await logout(request);
}

type CreateUserSessionOptions = {
    request: Request;
    userId: string;
    remember: boolean;
    redirectTo: string;
};
export async function createUserSession({
    request,
    userId,
    remember,
    redirectTo
}: CreateUserSessionOptions) {
    const session = await getSession(request);
    session.set(USER_SESSION_KEY, userId);
    return redirect(redirectTo, {
        headers: {
            'Set-Cookie': await sessionStorage.commitSession(session, {
                maxAge: remember
                    ? 60 * 60 * 24 * 7 // 7 days
                    : undefined
            })
        }
    });
}

export async function logout(request: Request) {
    const session = await getSession(request);
    return redirect('/', {
        headers: {
            'Set-Cookie': await sessionStorage.destroySession(session)
        }
    });
}
