define(['backbone', 'model/main-model', 'view/input-view',
				 'view/result-view', 'view/analysis-view'], function(Backbone, Main, InputView, ResultView, AnalysisView) {

	app = window.app = {};

	var model = app.model = new Main();
	app.inputView = new InputView({ model: model });
	app.resultView = new ResultView({ model: model });
	app.analysisView = new AnalysisView({ model: model });

});
