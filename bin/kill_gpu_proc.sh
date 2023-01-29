#!/usr/bin/env bash

ps -aux |grep train |awk '{print $2}'|xargs -n1 kill -9