import { Component, Input } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { API_URL } from '@shared/environment';
import { JobService } from '@shared/services/jobs.service';
import { AuthService } from '@shared/services/auth.service';
import { ToastrService } from 'ngx-toastr'
import { retry } from 'rxjs/operators';

@Component({
  selector: 'erosion-job-submission-button',
  template: `<button [class]="'btn' + disabled ? 'disabled' : ''" (click)="send($event)">Submit job</button>`,
})
export class ErosionJobSubmissionButtonComponent {
  @Input() jobId;
  @Input() disabled;

  constructor(private http: HttpClient, private jobService: JobService, private authService: AuthService, private toastr: ToastrService){}

  send() {
    if (!this.authService.getUser().permissions.includes('submit_jobs')) {
      this.toastr.error('Unauthorized')
      return
    }
    this.http.post(API_URL + '/jobs/submit/' + this.jobId).subscribe(
      output => {
        const jobToken = output.job_details.token;
        this.jobService.getResult({token: jobToken}).pipe(retry(10)).subscribe(
          data => {
            if (data.status === 'FAILED' || data.status === 'CANCELLED') return;
            const mostRecentAttempt = data.attempts.sort((a, b) => a.result_started > b.result_started ? -1 : 1)[0]
            const message = 'Erosion job succeeded.  Customer facing layer ' + mostRecentAttempt.result.overlays[0].name + ' created'
            this.toastr.success(message)
          }
        )
      }
    )
  }
}
