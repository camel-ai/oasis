import { describe, it, expect } from 'vitest';

import {
    singleLine,
    multiLine,
    pad,
    lpad,
    rpad,
    lrpad,
    makeBanner,
    makeTable
} from './string-utils';

describe('String Utilities', () => {
    describe('singleLine', () => {
        it('should join fragments with spaces', () => {
            expect(singleLine('hello', 'world')).toBe('hello world');
            expect(singleLine('one', 'two', 'three')).toBe('one two three');
            expect(singleLine('')).toBe('');
        });
    });

    describe('multiLine', () => {
        it('should join fragments with line breaks', () => {
            expect(multiLine('hello', 'world')).toBe('hello\nworld');
            expect(multiLine('one', 'two', 'three')).toBe('one\ntwo\nthree');
            expect(multiLine('')).toBe('');
        });
    });

    describe('pad', () => {
        it('should pad string on the left by default', () => {
            expect(pad('test', 8)).toBe('    test');
            expect(pad('test', 8, '-')).toBe('----test');
        });

        it('should pad string on the right when type is "right"', () => {
            expect(pad('test', 8, ' ', 'right')).toBe('test    ');
            expect(pad('test', 8, '-', 'right')).toBe('test----');
        });

        it('should pad string on both sides when type is "both"', () => {
            expect(pad('test', 8, ' ', 'both')).toBe('  test  ');
            expect(pad('test', 8, '-', 'both')).toBe('--test--');
        });

        it('should not pad if string is longer than target length', () => {
            expect(pad('testing', 4)).toBe('testing');
        });

        it('should only use first character of pad string', () => {
            expect(pad('test', 8, 'ab')).toBe('aaaatest');
        });

        it('should use space if pad string is empty', () => {
            expect(pad('test', 8, '')).toBe('    test');
        });
    });

    describe('lpad', () => {
        it('should pad string on the left', () => {
            expect(lpad('test', 8)).toBe('    test');
            expect(lpad('test', 8, '-')).toBe('----test');
        });
    });

    describe('rpad', () => {
        it('should pad string on the right', () => {
            expect(rpad('test', 8)).toBe('test    ');
            expect(rpad('test', 8, '-')).toBe('test----');
        });
    });

    describe('lrpad', () => {
        it('should pad string on both sides', () => {
            expect(lrpad('test', 8)).toBe('  test  ');
            expect(lrpad('test', 8, '-')).toBe('--test--');
        });
    });

    describe('makeBanner', () => {
        it('should create a banner with a message', () => {
            const banner = makeBanner('Hello World');
            expect(banner).toBeInstanceOf(Array);
            expect(banner.length).toBe(5);
            expect(banner[0]).toMatch(/^┌/); // starts with top-left corner
            expect(banner[2]).toContain('Hello World'); // contains message
            expect(banner[4]).toMatch(/┘$/); // ends with bottom-right corner
        });

        it('should create a banner with a message and title', () => {
            const banner = makeBanner('Hello World', 'Title');
            expect(banner).toBeInstanceOf(Array);
            expect(banner.length).toBe(7);
            expect(banner[1]).toContain('Title'); // contains title
            expect(banner[2]).toMatch(/^├/); // has divider after title
            expect(banner[4]).toContain('Hello World'); // contains message
        });

        it('should handle multi-line messages', () => {
            const banner = makeBanner(['Line 1', 'Line 2']);
            expect(banner).toBeInstanceOf(Array);
            expect(banner.length).toBe(6);
            expect(banner[2]).toContain('Line 1');
            expect(banner[3]).toContain('Line 2');
        });
    });

    describe('makeTable', () => {
        it('should create a table from object data', () => {
            const table = makeTable({ name: 'Test', value: 123 });
            expect(table).toBeInstanceOf(Array);
            expect(table.length).toBe(6);
            expect(table[2]).toContain('name: Test');
            expect(table[3]).toContain('value: 123');
        });

        it('should create a table with title', () => {
            const table = makeTable({ a: 1, b: 2 }, 'MyTable');
            expect(table).toBeInstanceOf(Array);
            expect(table.length).toBe(8);
            expect(table[1]).toContain('MyTable');
        });

        it('should align keys properly', () => {
            const table = makeTable({ short: 1, longerKey: 2 });
            // Find the line with 'short' and check that it's padded
            const shortLine = table.find(line => line.includes('short'));
            expect(shortLine).toContain('    short');
        });
    });
});
