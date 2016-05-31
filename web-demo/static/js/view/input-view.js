define(['template/input-template', 'backbone'], function(inputTemplate, Backbone) {

   var  InputView = Backbone.View.extend({

        events: {
            'click .btn': '_onAnswerClick'
        },

        initialize: function() {
            this.setElement("#input_container");
            this.render();
        },

        render: function() {
            var template = _.template( inputTemplate);
            this.$el.html(template);

            this.$story = this.$el.find('.story-text');
            this.$question = this.$el.find('.question-text');
        },

        _onAnswerClick: function () {
            var story = this.$story.val().trim(),
                question = this.$question.val().trim(),
                sentences = [];

            sentences = story.split('\n');

            this.model.set("story", sentences);
            this.model.set("question", question);

            if (!!story && !!question) {
                this.model.getAnswer();
            }

        }
    });

    return InputView;
});
