import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { BehaviorSubject } from 'rxjs';

@Injectable({
  providedIn: 'root',
})
export class LoanService {
  // Variables observables para cada clave del JSON
  private pagadoSubject = new BehaviorSubject<any>(null);
  private en_procesoSubject = new BehaviorSubject<any>(null);
  private incumplidoSubject = new BehaviorSubject<any>(null);
  private scoreSubject = new BehaviorSubject<any>(null);

  // Exponiendo como observables
  pagado$ = this.pagadoSubject.asObservable();
  en_proceso$ = this.en_procesoSubject.asObservable();
  incumplido$ = this.incumplidoSubject.asObservable();
  score$ = this.scoreSubject.asObservable();

  constructor(private http: HttpClient) {}

  // Método para enviar datos al endpoint y actualizar las variables observables
  submitLoanData(loanData: any) {
    const endpoint = 'http://127.0.0.1:4321/api/get-score'; // Cambia esta URL por el endpoint real

    this.http.post<any>(endpoint, loanData).subscribe({
      next: (response) => {
        // Actualiza las variables observables con las claves del JSON recibido
        let jsonResponse =
          typeof response === 'string' ? JSON.parse(response) : response;

        console.log(jsonResponse);
        // Actualiza las variables observables con las claves del JSON recibido
        this.pagadoSubject.next(jsonResponse.pagado);
        this.en_procesoSubject.next(jsonResponse.en_proceso);
        this.incumplidoSubject.next(jsonResponse.incumplido);
        this.scoreSubject.next(jsonResponse.score);

        // Console logs para verificar las actualizaciones
        console.log('pagado actualizado:', jsonResponse.pagado);
        console.log('en_proceso actualizado:', jsonResponse.en_proceso);
        console.log('incumplido actualizado:', jsonResponse.incumplido);
        console.log('score actualizado:', jsonResponse.score);
      },

      error: (error) => {
        console.error('Error al enviar datos del préstamo:', error);
        // En caso de error, resetea los valores de las variables observables
        this.pagadoSubject.next(null);
        this.en_procesoSubject.next(null);
        this.incumplidoSubject.next(null);
        this.scoreSubject.next(null);
      },
    });
  }
}
