import '@angular/localize/init';
import { platformBrowserDynamic } from '@angular/platform-browser-dynamic';

import { VERSION as CDK_VERSION } from '@angular/cdk';
import { VERSION as MAT_VERSION } from '@angular/material/core';

import { AppModule } from './app/app.module';
import { bootstrapApplication } from '@angular/platform-browser';
import { AppComponent } from './app/app.component';
import { provideHttpClient, withFetch } from '@angular/common/http';
import { provideAnimationsAsync } from '@angular/platform-browser/animations/async';

/* eslint-disable no-console */
console.info('Angular CDK version', CDK_VERSION.full);
console.info('Angular Material version', MAT_VERSION.full);


platformBrowserDynamic()
  .bootstrapModule(AppModule)
  .catch((err) => console.error(err));

bootstrapApplication(AppComponent, {
  providers: [provideHttpClient(withFetch()), provideAnimationsAsync('noop')],
}).catch((err) => console.error(err));

