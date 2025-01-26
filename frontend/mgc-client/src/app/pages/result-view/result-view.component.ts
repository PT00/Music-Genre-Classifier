import { Component } from '@angular/core';
import { ProbObjectService } from '../../services/prob-object.service';
import { ProbObject } from '../../models/predict-genre-response.model';
import { CommonModule } from '@angular/common';
import { PieChartComponent } from "../../components/pie-chart/pie-chart.component";

@Component({
  selector: 'app-result-view',
  standalone: true,
  imports: [CommonModule, PieChartComponent],
  templateUrl: './result-view.component.html',
  styleUrls: ['./result-view.component.scss'],
})
export class ResultViewComponent {
  probObject: ProbObject | null = null;
  pieChartData: { labels: string[]; values: number[] } | null = null;

  constructor(private probObjectService: ProbObjectService) {}

  ngOnInit() {
    this.probObject = this.probObjectService.getProbObject();
    console.log('Results: ', this.probObject);

    if (this.probObject) {
      this.pieChartData = {
        labels: Object.keys(this.probObject),
        values: Object.values(this.probObject),
      }
    } else {
      console.warn('No classification result found.');
    }
  }
}
