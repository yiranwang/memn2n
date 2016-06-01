define(['template/analysis-root-template', 'backbone', 'view/analysis-view-word', 'view/analysis-view'], function(analysisRootTemplate, Backbone, AnalysisWordView, AnalysisView) {

    var AnalysisRootView = Backbone.View.extend({
        initialize: function () {
            this.setElement("#analysis_container");
            this.render();
        },

        events: {
            'change select': '_onChangeModel'
        },

        render: function () {
            var $wordView = new AnalysisWordView({ model: this.model }).render().$el,
                $analysisView = new AnalysisView({ model: this.model }).render().$el;

            this.$el.html(_.template(analysisRootTemplate));

            this.$container = this.$el.find(".analysis-view-container");
            this.$container.append($analysisView);
            this.$container.append($wordView);

            this._views = {
                0: $analysisView,
                1: $wordView
            };

            this._currentModelIdx = 1;

            this._toggleView();

            return this;
        },

        _toggleView: function () {
            var viewToHide = this._currentModelIdx,
              viewToShow = this._currentModelIdx = (++this._currentModelIdx % 2);

            this._views[viewToHide].hide();
            this._views[viewToShow].show();
        },

        _onChangeModel: function () {
            this._toggleView();

            this.model.set({
                "answer": "",
                "correctAnswer": "",
                "answerProbability": 0,
                "memoryProbabilities": []
            });
        }
    });

    return AnalysisRootView;
});
