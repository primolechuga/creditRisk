import {
  AfterViewInit,
  Component,
  HostListener,
  OnInit,
  ViewChild,
} from '@angular/core';
import { MatSidenav } from '@angular/material/sidenav';
import { MessageService } from '../../services/Message/message.service';
import { BillComponent } from '../../components/bill/bill.component';
import { DataTableComponent } from '../../components/data-table/data-table.component';
import { AngularSplitModule } from 'angular-split';
import { AiChatComponent } from '../../components/ai-chat/ai-chat.component';
import { MatAccordion, MatExpansionModule } from '@angular/material/expansion';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { MatIconModule } from '@angular/material/icon';
import { MatToolbarModule } from '@angular/material/toolbar';
import { HeaderComponent } from '../../components/header/header.component';
import { TableService } from '../../services/TableService/table.service';
import { TreeDataModelComponent } from '../../components/tree-datamodel/tree-datamodel.component';
import { TreeChatComponent } from '../../components/tree-chat/tree-chat.component';
import { TreeReportsComponent } from '../../components/tree-reports/tree-reports.component';
import { ReportSheetComponent } from '../../components/report-sheet/report-sheet.component';
import { DomSanitizer, SafeUrl } from '@angular/platform-browser';
import { PrimeIcons, MenuItem } from 'primeng/api';

@Component({
  selector: 'app-data-model',
  standalone: true,
  imports: [
    CommonModule,
    AngularSplitModule,
    FormsModule,
    MatExpansionModule,
    MatToolbarModule,
    MatIconModule,
    DataTableComponent,
    AiChatComponent,
    HeaderComponent,
    TreeDataModelComponent,
    TreeChatComponent,
    TreeReportsComponent,
    ReportSheetComponent
  ],
  templateUrl: './datamodel.component.html',
  styleUrl: './datamodel.component.scss',
})

export class DataModelComponent implements OnInit, AfterViewInit {
  
  @ViewChild('sidenav', { static: false }) sidenav!: MatSidenav;
  @ViewChild('sidenavEnd', { static: false }) sidenavEnd!: MatSidenav;

  showLeftSidenav: boolean = true;
  showRightSidenav: boolean = true;
  mainArea: boolean = true;

  isMobile: boolean = false; // Para saber si la pantalla es pequeña
  leftPanelSize: number = 23; // Tamaño predeterminado del panel izquierdo
  rightPanelSize: number = 77; // Tamaño predeterminado del panel derecho

  tablesNames: any;

  columns: any[] = [];

  first = 0;
  rows = 8;
  loading: boolean = true;

  products!: any[];

  tableData: any[] = [];

  menu: any;
  dataModelNodes: string[] = ['Modelo de datos'];

  activeTab: string = 'datamodel';

  chats: any[] = [];

  constructor(
    private messageService: MessageService,
    private tableService: TableService
  ) {
    
  }


  ngOnInit() {
    this.checkMobileResolution();
    this.loading = false;
    this.updateNavContainerHeight();

    this.tableService.menu$.subscribe((response) => {
      this.menu = response;
    });

    this.getTableNames();
  }


  ngAfterViewInit(): void {}


  checkMobileResolution() {
    if (typeof window !== 'undefined') {
      
      const screenWidth = window.innerWidth;
      
      if (screenWidth <= 766) {
        
        this.isMobile = true;
        this.leftPanelSize = 0;
        this.rightPanelSize = 100;
      } 
      
      else {
        
        this.isMobile = false;
        this.leftPanelSize = 23;
        this.rightPanelSize = 77;
      }
    }
  }

  updateResolution() {
    const screenWidth = window.innerWidth;
    this.isMobile = screenWidth <= 766;

    // Restablecer los tamaños de los paneles para resoluciones grandes
    if (!this.isMobile) {
      this.leftPanelSize = 23; // Ajusta el tamaño del panel izquierdo
      this.rightPanelSize = 77; // Ajusta el tamaño del panel derecho
      this.showLeftSidenav = true; // Mostrar el menú izquierdo en pantallas grandes
      this.mainArea = true;
    } else {
      this.leftPanelSize = 0; // El panel izquierdo debe ocupar 0 espacio en pantallas pequeñas
      this.rightPanelSize = 100; // El panel derecho ocupa todo el espacio
      this.showLeftSidenav = false; // El menú izquierdo se oculta en pantallas pequeñas
    }


    this.updateNavContainerHeight();
  }


  updateNavContainerHeight() {
    if (typeof window !== 'undefined' && typeof document !== 'undefined') {
      const navContainerHeight = window.innerHeight - 60; // Get the current viewport height - 60
      document.documentElement.style.setProperty(
        '--nav-container-height',
        `${navContainerHeight}px`
      );
    }
  
  }


  @HostListener('window:resize', ['$event'])
  onResize(event: any) {
    this.updateNavContainerHeight();
    this.updateResolution();
  }


  toggleMenu() {
    this.showLeftSidenav = !this.showLeftSidenav;
  }


  onTabChanged(tab: string): void {
    this.activeTab = tab;
    this.menu = 'Nothing';
    console.log(this.tableData)
  }


  getTableNames(): void {
    this.tableService.getTableNames().subscribe({
      next: (tableNames) => {
        // Extraer solo los nombres de las tablas
        this.tablesNames = Object.keys(tableNames);

        // Agregar los nombres de las tablas a dataModelNodes
        this.tablesNames.forEach((table: string) => {
          this.dataModelNodes.push(table);
        });
      },
      error: (error) => {
        console.error(error);
      },
    });
  }
  
}
