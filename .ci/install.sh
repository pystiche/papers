#!/usr/bin/env sh

pip3 install -r requirements-test.txt

pip3 install --upgrade .  codecov

pip3 list
