import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ChartModule } from 'primeng/chart';
import { ProgressBarModule } from 'primeng/progressbar';
import { LoanService } from '../../services/loan-service/loan.service'; // Importa el servicio LoanService
import { Subscription } from 'rxjs';

@Component({
  selector: 'app-score-card',
  standalone: true,
  imports: [CommonModule, ChartModule, ProgressBarModule],
  templateUrl: './scorecard.component.html',
  styleUrls: ['./scorecard.component.scss'],
})
export class ScoreCardComponent implements OnInit {
  pieChartData = {
    labels: ['Current', 'Fully Paid', 'Failure'],
    datasets: [
      {
        data: [64.04, 23.41, 12.55], // Ejemplo de datos
        backgroundColor: ['#FF6384', '#36A2EB', '#FFCE56'],
      },
    ],
  };

  pieChartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
    },
  };

  paid: number = 0;
  current: number = 0;
  failure: number = 0;
  score: number = 0;

  private subscriptions: Subscription[] = [];

  constructor(private loanService: LoanService) {}

  ngOnInit(): void {
    this.subscriptions.push(
      this.loanService.pagado$.subscribe(
        (data) => (this.paid = data * 100 || 0)
      ),
      this.loanService.en_proceso$.subscribe(
        (data) => (this.current = data * 100 || 0)
      ),
      this.loanService.incumplido$.subscribe(
        (data) => (this.failure = data * 100 || 0)
      ),
      this.loanService.score$.subscribe((data) => (this.score = data || 0))
    );
  }

  ngOnDestroy(): void {
    this.subscriptions.forEach((sub) => sub.unsubscribe());
  }
}
