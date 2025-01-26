import { ComponentFixture, TestBed } from '@angular/core/testing';

import { DataModelComponent } from './datamodel.component';

describe('DataModel', () => {
  let component: DataModelComponent;
  let fixture: ComponentFixture<DataModelComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [DataModelComponent],
    }).compileComponents();

    fixture = TestBed.createComponent(DataModelComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
