// Netlify Function: the whole JSON API (/api/*), backed by Netlify Blobs.
import { getStore } from '@netlify/blobs';
import { netlifyHandler } from '../../lib/netlify.js';

export default netlifyHandler(
  (name) => getStore({ name, consistency: 'strong' }),
  (key) => Netlify.env.get(key),
);

export const config = { path: '/api/*' };
