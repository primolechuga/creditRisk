
FROM node:alpine

WORKDIR /usr/src/app

COPY . .

RUN npm install -g @angular/cli

RUN npm install

EXPOSE 4000

RUN npm run build

CMD ["npm", "run" ,"ng", "serve"]