define(['backbone', 'model/main-model', 'view/input-view', 'view/result-view', 'view/analysis-view'], function(Backbone, Main, InputView, ResultView, AnalysisView) {

	app = window.app = {};

	app.model = new Main();
	app.inputView = new InputView();
	app.resultView = new ResultView();
	app.analysisView = new AnalysisView();

});
