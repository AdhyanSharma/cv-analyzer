FROM node:20-alpine AS build

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .

RUN npm run build


FROM nginx:alpine

COPY nginx.conf.template /etc/nginx/templates/default.conf.template
COPY render-nginx-entrypoint.sh /usr/local/bin/render-nginx-entrypoint.sh
RUN chmod +x /usr/local/bin/render-nginx-entrypoint.sh

COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

ENTRYPOINT ["/usr/local/bin/render-nginx-entrypoint.sh"]
CMD ["nginx", "-g", "daemon off;"]
