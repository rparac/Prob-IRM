ENV="${ENV:-buttons}"
AGENT_TYPE="${AGENT_TYPE:-rm}"
SEED="${SEED:-123}"
FOLDER="${FOLDER:-debug}"

if [ $ENV = "buttons" ]
then
    python \
        run.py \
        run=$AGENT_TYPE \
        run.training=True \
        run.name=$ENV-$AGENT_TYPE-$SEED \
        run.log_dir=logs/$FOLDER/$AGENT_TYPE \
        run.seed=$SEED \
        env=$ENV \
        env/$ENV/agents/agent_type@env._agent1_config.agent=$AGENT_TYPE \
        env/$ENV/agents/agent_type@env._agent2_config.agent=$AGENT_TYPE \
        env/$ENV/agents/agent_type@env._agent3_config.agent=$AGENT_TYPE \
        "$@"
else
    python \
        run.py \
        run=$AGENT_TYPE \
        run.training=True \
        run.name=$ENV-$AGENT_TYPE-$SEED \
        run.log_dir=logs/$FOLDER/$AGENT_TYPE \
        run.seed=$SEED \
        env=$ENV \
        env/$ENV/agents/agent_type@env._agent1_config.agent=$AGENT_TYPE \
        env/$ENV/agents/agent_type@env._agent2_config.agent=$AGENT_TYPE \
        "$@"
fi

