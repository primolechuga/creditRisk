import { Component } from '@angular/core';
import { InputTextModule } from 'primeng/inputtext';
import { DropdownModule } from 'primeng/dropdown';
import { CalendarModule } from 'primeng/calendar';
import { ButtonModule } from 'primeng/button';
import { CardModule } from 'primeng/card';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';

import { LoanService } from '../../services/loan-service/loan.service';
import { ScoreCardComponent } from '../scorecard/scorecard.component';

@Component({
  selector: 'loan-form',
  templateUrl: './loan-form.component.html',
  styleUrls: ['./loan-form.component.scss'],
  standalone: true,
  imports: [
    InputTextModule,
    DropdownModule,
    CalendarModule,
    ButtonModule,
    CardModule,
    FormsModule,
    ReactiveFormsModule,
    CommonModule,
    ScoreCardComponent,
  ],
})
export class LoanFormComponent {
  loanAmnt!: number;
  fundedAmnt!: number;
  fundedAmntInv!: number;
  term!: number;
  intRate!: number;
  grade!: string;
  homeOwnership!: string;
  verificationStatus!: string;
  purpose!: string;
  dti!: number;
  inqLast6mths!: number;
  openAcc!: number;
  revolUtil!: number;
  outPrncp!: number;
  totalPymnt!: number;
  totalRecInt!: number;
  lastPymntAmnt!: number;
  totCurBal!: number;
  totalRevHiLim!: number;
  mthsSinceIssueD!: Date | null;
  mthsSinceLastPymntD!: Date | null;
  mthsSinceLastCreditPullD!: Date | null;

  constructor(private loanService: LoanService) {}

  gradeOptions = [
    { label: 'A', value: 'A' },
    { label: 'B', value: 'B' },
    { label: 'C', value: 'C' },
    { label: 'D', value: 'D' },
    { label: 'E', value: 'E' },
  ];
  homeOwnershipOptions = [
    { label: 'Own', value: 'OWN' },
    { label: 'Rent', value: 'RENT' },
    { label: 'Mortage', value: 'MORTAGE' },
    { label: 'Other', value: 'OTHER' },
    { label: 'None', value: 'NONE' },
  ];
  verificationStatusOptions = [
    { label: 'Verified', value: 'Verified' },
    { label: 'Source Verified', value: 'Source Verified' },
    { label: 'Not Verified', value: 'Not Verified' },
  ];

  purposeOptions = [
    { label: 'Debt Consolidation', value: 'debt_consolidation' },
    { label: 'Credit Card', value: 'credit_card' },
    { label: 'Home Improvement', value: 'home_improvement' },
    { label: 'Other', value: 'other' },
    { label: 'Major Purchase', value: 'major_purchase' },
  ];

  onSubmit(form: any) {
    // Crear un objeto con los datos del formulario
    const loanData = {
      loan_amnt: this.loanAmnt,
      funded_amnt: this.fundedAmnt,
      funded_amnt_inv: this.fundedAmntInv,
      term: this.term,
      int_rate: this.intRate,
      grade: this.grade,
      home_ownership: this.homeOwnership,
      verification_status: this.verificationStatus,
      purpose: this.purpose,
      dti: this.dti,
      inq_last_6mths: this.inqLast6mths,
      open_acc: this.openAcc,
      revol_util: this.revolUtil,
      out_prncp: this.outPrncp,
      total_pymnt: this.totalPymnt,
      total_rec_int: this.totalRecInt,
      last_pymnt_amnt: this.lastPymntAmnt,
      tot_cur_bal: this.totCurBal,
      total_rev_hi_lim: this.totalRevHiLim,
      mths_since_issue_d: this.mthsSinceIssueD,
      mths_since_last_pymnt_d: this.mthsSinceLastPymntD,
      mths_since_last_credit_pull_d: this.mthsSinceLastCreditPullD,
    };

    // Llamar al servicio para enviar los datos
    this.loanService.submitLoanData(loanData);

    // Mostrar los datos del formulario para depuración (opcional)
    console.log('Formulario enviado:', loanData);
  }
}
