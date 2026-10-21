// Netlify Function: the whole JSON API (/api/*), backed by Netlify Blobs.
import { getStore } from '@netlify/blobs';
import { netlifyHandler } from '../../lib/netlify.js';

export default netlifyHandler(() => getStore({ name: 'lemma', consistency: 'strong' }));

export const config = { path: '/api/*' };
