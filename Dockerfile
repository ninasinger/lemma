FROM node:22-alpine
WORKDIR /app
COPY package.json server.js ./
COPY lib ./lib
COPY scripts/seed.js ./scripts/seed.js
COPY public ./public
RUN mkdir -p /data && chown node:node /data
ENV NODE_ENV=production HOST=0.0.0.0 PORT=3000 TRUST_PROXY=1 DATA_FILE=/data/db.json
VOLUME /data
EXPOSE 3000
USER node
CMD ["node", "server.js"]
